# Code generation context: Thursday, April 17, 2025 at 7:21:43 AM CDT
# Location context: Murfreesboro, Tennessee, United States

import os
import json
import networkx as nx
import google.generativeai
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (LSTM, Dense, Input, TimeDistributed, Attention,
                                     Concatenate, Flatten, Reshape, RepeatVector,
                                     MultiHeadAttention, LayerNormalization, Dropout)
from tensorflow.keras.models import Model
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import google.cloud.pubsub_v1 as pubsub
from datetime import datetime, timedelta
from google.cloud import storage
import logging # Import the logging module
import time # For retry delays
import sys # For setting up logging output to stdout
import math
import random
from collections import namedtuple, deque # Added deque for history

# Optional, but highly recommended for RRT* performance:
try:
    from scipy.spatial import KDTree
    USE_KDTREE = True
except ImportError:
    USE_KDTREE = False
    # logger is not defined yet, print warning here
    print("Warning: scipy.spatial.KDTree not found. Falling back to slower nearest neighbor search.")

# Import specific exceptions for Google Cloud APIs
from google.api_core import exceptions as google_api_exceptions
from google.cloud import exceptions as google_cloud_exceptions

# --- Logging Configuration ---
# Configure logging once at the start of the application
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%dT%H:%M:%S%z') # ISO 8601 Format
log_level = logging.INFO # Use INFO for general operation, DEBUG for detailed development info

# Console Handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)

# File Handler (using RotatingFileHandler for better log management)
log_file = 'robot_app.log'
try:
    # Use RotatingFileHandler for long-running processes
    from logging.handlers import RotatingFileHandler
    # Example: 5MB files, keep 3 backup logs
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
    file_handler.setFormatter(log_formatter)
except ImportError:
    print("Warning: RotatingFileHandler not available. Using basic FileHandler.")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)


# Get the root logger and add handlers
# Avoid adding handlers multiple times if this script is re-run in an interactive session
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

# Get a logger for this specific module (used in top-level functions)
logger = logging.getLogger(__name__)

# --- Configuration ---
logger.info("Loading configuration...")
# TODO: Move API keys, paths, and IDs to environment variables or a dedicated config file/service
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE") # Example using env var
YOLO_CFG_PATH = "yolov3.cfg"
YOLO_WEIGHTS_PATH = "yolov3.weights"
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", 'your-gcp-project-id') # Example
PUBSUB_TOPIC_ID = 'object_exploration'
PUBSUB_SUBSCRIPTION_ID = 'robot_subscription'
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'your-bucket-name') # Example
GCS_BLOB_NAME = 'graph_data.json'

# RRT* Planner Configuration
CONFIG_SPACE_BOUNDS = (0, 20, 0, 15) # Example: (min_x, max_x, min_y, max_y)
STEP_SIZE = 0.5
MAX_ITERATIONS = 5000
GOAL_SAMPLE_RATE = 0.1
SEARCH_RADIUS = 1.5
ROBOT_RADIUS = 0.3

# --- Configuration Validation (Basic Example) ---
if GEMINI_API_KEY == "YOUR_API_KEY_HERE":
    logger.warning("GEMINI_API_KEY is not set or using placeholder. Set environment variable.")
if GCP_PROJECT_ID == 'your-gcp-project-id':
    logger.warning("GCP_PROJECT_ID is not set. Set environment variable or replace placeholder.")
if GCS_BUCKET_NAME == 'your-bucket-name':
    logger.warning("GCS_BUCKET_NAME is not set. Set environment variable or replace placeholder.")
if not os.path.exists(YOLO_CFG_PATH):
     logger.error("YOLO config file not found at: %s", YOLO_CFG_PATH)
     # Decide if fatal: sys.exit(1)
if not os.path.exists(YOLO_WEIGHTS_PATH):
     logger.error("YOLO weights file not found at: %s", YOLO_WEIGHTS_PATH)
     # Decide if fatal: sys.exit(1)


# --- Configure Gemini API ---
try:
    # Only configure if key seems valid
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
        google.generativeai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured.")
    else:
        logger.error("Skipping Gemini API configuration due to missing/placeholder key.")
except Exception as e:
    logger.critical("Failed to configure Gemini API: %s", e, exc_info=True)
    # Decide if fatal depending on whether Gemini is essential

# --- Object Detection ---
logger.info("Loading Object Detector (YOLO)...")
object_detector = None
try:
    object_detector = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    # Optional: Set preferable backend and target
    # object_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # object_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    logger.info("Object Detector loaded successfully.")
except cv2.error as e:
    logger.critical("Failed to load YOLO model from %s, %s: %s", YOLO_CFG_PATH, YOLO_WEIGHTS_PATH, e, exc_info=True)
    sys.exit(1) # Exit if detector is essential
except FileNotFoundError as e:
     logger.critical("YOLO config/weights file not found: %s", e, exc_info=True)
     sys.exit(1) # Exit if detector is essential


# --- Kalman Filter ---
KALMAN_STATE_DIM = 4 # Using 4D state [x, y, vx, vy] to match TF models
KALMAN_MEASURE_DIM = 2 # Measuring [x, y]

class KalmanTracker:
    def __init__(self, initial_state):
        self.logger = logging.getLogger(__name__)
        if initial_state is None or initial_state.shape[0] != KALMAN_STATE_DIM:
             msg = f"Initial state must have dimension {KALMAN_STATE_DIM}, got shape {initial_state.shape if initial_state is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)

        # IMPORTANT: Matrices below NEED TUNING based on actual system dynamics and noise.
        self.filter = cv2.KalmanFilter(KALMAN_STATE_DIM, KALMAN_MEASURE_DIM)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        dt = 1.0 # Assuming dt=1 for simplicity, MUST replace with actual time delta
        self.filter.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.filter.processNoiseCov = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.03 # Tune this
        self.filter.measurementNoiseCov = np.eye(KALMAN_MEASURE_DIM, dtype=np.float32) * 0.1 # Tune this (sensor noise)
        self.filter.statePost = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.statePre = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.errorCovPost = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune initial uncertainty
        self.filter.errorCovPre = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune initial uncertainty

    def predict(self):
        try:
            predicted_state = self.filter.predict()
            return predicted_state
        except cv2.error as e:
            self.logger.error("OpenCV Kalman predict error: %s", e, exc_info=True)
            return None # Or return previous state?

    def update(self, measurement):
        if measurement is None or measurement.shape[0] != KALMAN_MEASURE_DIM:
             msg = f"Measurement must have dimension {KALMAN_MEASURE_DIM}, got shape {measurement.shape if measurement is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)
        measurement_col = measurement.reshape(-1, 1).astype(np.float32)
        try:
            self.filter.correct(measurement_col)
            return self.filter.statePost
        except cv2.error as e:
            self.logger.error("OpenCV Kalman correct error: %s", e, exc_info=True)
            return self.filter.statePost # Return previous state on update failure


# --- Transformer Model (TensorFlow) ---
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        # Ensure key_dim is compatible or let MultiHeadAttention handle it if possible
        # MHA requires input dim last dimension to be divisible by num_heads if key_dim not set
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads if self.d_model % self.num_heads == 0 else self.d_model)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [Dense(self.dff, activation='relu'), Dense(self.d_model)]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, x, training, mask=None):
        attn_output, attn_weights = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class ObjectBehaviorLearner:
    def __init__(self, prediction_horizon=10, state_dimension=KALMAN_STATE_DIM, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(__name__)
        self.prediction_horizon = prediction_horizon
        self.state_dimension = state_dimension
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.object_graph = nx.Graph()

        try:
            if self.d_model % self.num_heads != 0:
                 self.logger.warning("Transformer d_model (%d) is not divisible by num_heads (%d). MultiHeadAttention might behave unexpectedly or use approximation.", self.d_model, self.num_heads)

            input_layer = Input(shape=(None, self.state_dimension), name="transformer_input")
            x = Dense(d_model, name="input_embedding")(input_layer)
            # TODO: Add Positional Encoding if sequence order matters significantly
            for i in range(num_layers):
                x = TransformerBlock(d_model, num_heads, dff, name=f"transformer_block_{i}")(x, training=True) # Placeholder training=True, needs to be passed correctly

            output_layer = TimeDistributed(Dense(self.state_dimension), name="output_dense")(x)
            self.model = Model(inputs=input_layer, outputs=output_layer)
            self.model.compile(optimizer='adam', loss='mse')
            self.logger.info("Transformer Model compiled successfully.")
            # self.model.summary(print_fn=self.logger.info) # Log summary
        except Exception as e:
            self.logger.critical("Failed to build or compile Transformer model: %s", e, exc_info=True)
            raise # Re-raise exception as model is critical


    def train(self, past_states, target_states):
        self.logger.info("Starting Transformer training...")
        try:
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0) # verbose=0 for less console spam
            self.logger.info("Transformer training finished. Final validation loss: %s", history.history['val_loss'][-1])
        except tf.errors.InvalidArgumentError as e:
            self.logger.error("TensorFlow InvalidArgumentError during training (check input shapes/types): %s", e, exc_info=True)
        except Exception as e:
            self.logger.error("Unexpected error during Transformer training: %s", e, exc_info=True)


    def predict(self, past_states):
        self.logger.debug("Predicting with Transformer for %d samples", past_states.shape[0])
        try:
            predictions = self.model.predict(past_states)
            return predictions
        except tf.errors.InvalidArgumentError as e:
            self.logger.error("TensorFlow InvalidArgumentError during prediction (check input shape/type): %s", e, exc_info=True)
            return None
        except Exception as e:
             self.logger.error("Unexpected error during Transformer prediction: %s", e, exc_info=True)
             return None

    def update_graph(self, object_id, state, predicted_state):
        # ... (Keep graph update logic from previous version) ...
        # Consider adding logging for node/edge updates
        now = datetime.utcnow()
        state_values = state.flatten()
        predicted_state_values = predicted_state.flatten()

        if object_id not in self.object_graph.nodes:
            self.object_graph.add_node(object_id, state=state_values.tolist(), label=None, description=None, functionality=None, affordances=None, gemini_confidence=None, explored=False, timestamp=now.isoformat())
            # self.logger.debug("Added node %s to graph", object_id)
        else:
            self.object_graph.nodes[object_id]['state'] = state_values.tolist()
            self.object_graph.nodes[object_id]['timestamp'] = now.isoformat()

        error = np.linalg.norm(state_values - predicted_state_values)
        self.object_graph.add_edge(object_id, object_id, weight=error, type='prediction_error')


    # --- Pub/Sub Methods ---
    # Modified to accept publisher client and topic path
    def publish_object_info(self, node_id, publisher, topic_path):
         if not publisher or not topic_path:
              self.logger.warning("Cannot publish object info: Publisher or topic_path not available.")
              return
         unexplored_objects = self.get_unexplored_objects()
         if unexplored_objects:
             message = {'node_id': node_id, 'unexplored_objects': unexplored_objects}
             data = json.dumps(message).encode('utf-8')
             try:
                 future = publisher.publish(topic_path, data)
                 self.logger.info("Published object info message ID: %s", future.result())
             except google_api_exceptions.GoogleAPIError as e:
                 self.logger.error("Pub/Sub API Error publishing object info: %s", e, exc_info=True)
             except Exception as e:
                 self.logger.error("Unexpected error publishing object info: %s", e, exc_info=True)

    def request_object_info(self, node_id, publisher, topic_path):
         if not publisher or not topic_path:
              self.logger.warning("Cannot request object info: Publisher or topic_path not available.")
              return
         message = {'node_id': node_id, 'requesting_object_info': True}
         data = json.dumps(message).encode('utf-8')
         try:
             future = publisher.publish(topic_path, data)
             self.logger.info("Published object info request message ID: %s", future.result())
         except google_api_exceptions.GoogleAPIError as e:
              self.logger.error("Pub/Sub API Error publishing object info request: %s", e, exc_info=True)
         except Exception as e:
              self.logger.error("Unexpected error publishing object info request: %s", e, exc_info=True)

    def get_unexplored_objects(self):
         # ... (keep implementation) ...
         return [node for node, data in self.object_graph.nodes(data=True) if not data.get('explored', False)]


# --- Prediction Combiner (TensorFlow) ---
class PredictionCombiner(tf.keras.Model):
    def __init__(self, transformer_output_dim, context_dim, output_dim=KALMAN_STATE_DIM, prediction_horizon=10, use_confidence_weighting=True, dropout_rate=0.2):
        super(PredictionCombiner, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.use_confidence_weighting = use_confidence_weighting
        self.dropout_rate = dropout_rate
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim

        # TODO: Define input layer explicitly to calculate concatenated size accurately
        # For now, define dense layers with reasonable sizes
        self.dense1 = Dense(128, activation='relu', name="combiner_dense1")
        self.dense2 = Dense(64, activation='relu', name="combiner_dense2")
        self.dense3 = Dense(self.output_dim * self.prediction_horizon, name="combiner_output_dense")
        self.layer_norm = LayerNormalization(epsilon=1e-6, name="combiner_layernorm")
        self.dropout = Dropout(self.dropout_rate, name="combiner_dropout")
        self.logger.info("Prediction Combiner Initialized.")

    def call(self, inputs, training=None):
        try:
            transformer_predictions, transformer_confidences, environment_info, robot_state, obstacles = inputs

            # Flattening - Check dimensions carefully
            transformer_predictions = Flatten()(transformer_predictions) if len(tf.shape(transformer_predictions)) > 2 else transformer_predictions
            transformer_confidences = Flatten()(transformer_confidences) if len(tf.shape(transformer_confidences)) > 1 else transformer_confidences
            environment_info = Flatten()(environment_info) if len(tf.shape(environment_info)) > 2 else environment_info
            robot_state = Flatten()(robot_state) if len(tf.shape(robot_state)) > 2 else robot_state
            obstacles = Flatten()(obstacles) if len(tf.shape(obstacles)) > 1 else obstacles

            if self.use_confidence_weighting and transformer_confidences is not None:
                confidences_exp = tf.expand_dims(transformer_confidences, axis=-1)
                transformer_predictions = transformer_predictions * confidences_exp

            # TODO: Ensure all inputs are correctly shaped numerical tensors before concatenating
            combined_input = Concatenate()([transformer_predictions, environment_info, robot_state, obstacles])

            combined_input = self.layer_norm(combined_input, training=training)
            combined_input = self.dropout(combined_input, training=training)
            combined_input = self.dense1(combined_input)
            combined_input = self.dense2(combined_input)
            waypoints_flat = self.dense3(combined_input)

            waypoints = Reshape((self.prediction_horizon, self.output_dim))(waypoints_flat)
            return waypoints
        except tf.errors.InvalidArgumentError as e:
             self.logger.error("TF InvalidArgumentError in Combiner call (check input shapes/types): %s", e, exc_info=True)
             return None # Indicate failure
        except Exception as e:
             self.logger.error("Unexpected error in PredictionCombiner call: %s", e, exc_info=True)
             return None # Indicate failure


# --- ROS2 Node ---
environment_info_global = None # Still using global, consider alternatives

class EnvironmentInfoNode(Node):
    def __init__(self):
        super().__init__('environment_info_node')
        # Get logger provided by Node class
        self.logger = self.get_logger()
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, 10)
        self.logger.info("EnvironmentInfoNode created and subscribed to /scan.")

    def lidar_callback(self, msg):
        global environment_info_global
        try:
            # TODO: Implement robust Lidar processing
            ranges = np.array(msg.ranges)
            finite_ranges = ranges[np.isfinite(ranges)]
            if len(finite_ranges) > 0:
                 min_range = np.min(finite_ranges)
                 mean_range = np.mean(finite_ranges)
                 # Example processed info
                 processed_info = np.array([min_range, mean_range], dtype=np.float32)
            else:
                 processed_info = np.array([-1.0, -1.0], dtype=np.float32) # Indicate no valid ranges

            environment_info_global = processed_info
            # self.logger.debug(f'Lidar processed: {environment_info_global}') # DEBUG level
        except Exception as e:
            # Log error using the node's logger
            self.logger.error("Error processing lidar scan: %s", e, exc_info=True)
            # Decide how to handle global state on error (e.g., set to None?)
            # environment_info_global = None


# --- Cloud Storage Function ---
# (Keep function definition from previous version with logging)
def save_graph_to_cloud_storage(graph, bucket_name, blob_name):
    """Saves the object graph (NetworkX) to Google Cloud Storage as JSON."""
    func_logger = logging.getLogger(__name__) # Get logger
    storage_client = None
    try:
        storage_client = storage.Client(project=GCP_PROJECT_ID)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        graph_data = nx.node_link_data(graph)
        graph_json = json.dumps(graph_data, indent=2)
        blob.upload_from_string(graph_json, content_type='application/json')
        func_logger.info("Graph data successfully saved to gs://%s/%s", bucket_name, blob_name)
    # Specific exceptions first
    except google_cloud_exceptions.NotFound as e:
        func_logger.error("GCS Error: Bucket '%s' not found or blob path issue: %s", bucket_name, e)
    except google_cloud_exceptions.Forbidden as e:
        func_logger.error("GCS Error: Permission denied for bucket '%s' or blob '%s': %s", bucket_name, blob_name, e)
    # Catch other Google API errors
    except google_api_exceptions.GoogleAPIError as e:
        func_logger.error("GCS API Error saving graph: %s", e, exc_info=True)
    # Catch other potential errors (network, JSON serialization, etc.)
    except Exception as e:
        func_logger.error("Unexpected error saving graph to GCS: %s", e, exc_info=True)


# --- Robot Navigator (Gemini Integration) ---
# (Keep RobotNavigator class definition from previous version with logging & retries)
class RobotNavigator:
    def __init__(self, object_behavior_learner=None, max_retries=3, initial_delay=1):
        self.logger = logging.getLogger(__name__)
        self.object_learner = object_behavior_learner
        self.robot_state = None
        self.environment_info = None
        self.gemini_model = None
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
                self.gemini_model = google.generativeai.GenerativeModel('gemini-pro')
                self.logger.info("Gemini Model (%s) initialized.", 'gemini-pro')
            else:
                 self.logger.error("Gemini API Key is missing or placeholder. Cannot initialize Gemini Model.")
        except Exception as e:
            self.logger.error("Error initializing Gemini Model: %s", e, exc_info=True)
            self.gemini_model = None

    def update_context(self, robot_state, environment_info):
        self.robot_state = robot_state
        self.environment_info = environment_info

    def format_environment_info(self):
        if self.environment_info is None: return "No environment information available."
        try:
            # TODO: Implement better formatting based on actual data structure
            return ", ".join(map(str, self.environment_info))
        except Exception as e:
            self.logger.error("Error formatting environment info: %s", e, exc_info=True)
            return "Error formatting environment data."

    def format_graph_info(self):
        if self.object_learner is None or self.object_learner.object_graph is None: return "No object graph available."
        try:
            # TODO: Consider pruning/summarizing large graphs
            graph_data = nx.node_link_data(self.object_learner.object_graph)
            return json.dumps(graph_data, indent=2)
        except Exception as e:
            self.logger.error("Error formatting graph info: %s", e, exc_info=True)
            return "Error formatting graph data."

    def _build_gemini_prompt(self, user_command):
        try:
            env_info_str = self.format_environment_info()
            graph_info_str = self.format_graph_info()
            robot_state_str = str(self.robot_state)
        except Exception as e:
             self.logger.error("Error formatting context for Gemini prompt: %s", e, exc_info=True)
             env_info_str = "Error formatting environment data."
             graph_info_str = "Error formatting graph data."
             robot_state_str = "Error formatting robot state."
        # Prompt structure from before
        prompt = f"""
        You are a robot navigation assistant... [Rest of prompt details: Role, Task, Context, Instructions, Output Format Example]...

        Context:
        Robot State: {robot_state_str}
        Environment Info (e.g., sensor data): {env_info_str}
        Object Graph (Known objects, their states, and properties):
        ```json
        {graph_info_str}
        ```

        User Command: "{user_command}"

        Instructions:
        Provide a sequence of high-level actions... [Rest of prompt details]...
        """
        return prompt

    def get_gemini_navigation_instructions(self, user_command):
        """Queries Gemini for navigation instructions with retries."""
        if not self.gemini_model:
            self.logger.error("Cannot get Gemini instructions: Model not available.")
            return None
        prompt = self._build_gemini_prompt(user_command)
        self.logger.info("Querying Gemini with user command: '%s'", user_command)
        self.logger.debug("Gemini Prompt Length: %d chars", len(prompt)) # Log length instead of full prompt often

        retries = 0
        delay = self.initial_delay
        while retries < self.max_retries:
            try:
                response = self.gemini_model.generate_content(prompt)
                # TODO: Check response.prompt_feedback for safety blocks
                if response.prompt_feedback.block_reason:
                     self.logger.error("Gemini request blocked due to safety settings: %s", response.prompt_feedback.block_reason)
                     return None # Blocked, no point retrying?
                navigation_instructions_text = response.text
                self.logger.info("Received Gemini response.")
                return navigation_instructions_text

            except google_api_exceptions.ResourceExhausted as e:
                self.logger.warning("Gemini API Error (ResourceExhausted/429 - Rate Limit?): %s. Retrying (%d/%d) in %d s...", e, retries + 1, self.max_retries, delay)
            except google_api_exceptions.InternalServerError as e: # 500
                self.logger.warning("Gemini API Error (InternalServerError/500): %s. Retrying (%d/%d) in %d s...", e, retries + 1, self.max_retries, delay)
            except google_api_exceptions.ServiceUnavailable as e: # 503
                self.logger.warning("Gemini API Error (ServiceUnavailable/503): %s. Retrying (%d/%d) in %d s...", e, retries + 1, self.max_retries, delay)
            except google_api_exceptions.DeadlineExceeded as e: # 504 / Timeout
                 self.logger.warning("Gemini API Error (DeadlineExceeded/Timeout): %s. Retrying (%d/%d) in %d s...", e, retries + 1, self.max_retries, delay)

            except google_api_exceptions.PermissionDenied as e: # 403
                self.logger.error("Gemini API Error (PermissionDenied/403): Check API Key permissions. %s", e, exc_info=True)
                return None
            except google_api_exceptions.InvalidArgument as e: # 400
                 self.logger.error("Gemini API Error (InvalidArgument/400): Check prompt format/content. %s", e, exc_info=True)
                 return None
            except google_api_exceptions.GoogleAPIError as e:
                self.logger.error("Unhandled Gemini API Error: %s", e, exc_info=True)
                return None
            except Exception as e: # Catch other potential errors (network, library)
                self.logger.error("Unexpected error calling Gemini API: %s", e, exc_info=True)
                return None

            time.sleep(delay)
            retries += 1
            delay *= 2 # Exponential backoff

        self.logger.error("Gemini API call failed after %d retries.", self.max_retries)
        return None

    def parse_gemini_instructions(self, instructions_text):
        # (Keep implementation from previous version with logging)
        if not instructions_text: return None
        try:
            # Improved regex to handle potential leading/trailing whitespace
            match = re.search(r'```json\s*([\s\S]*?)\s*```', instructions_text, re.MULTILINE)
            if match:
                json_str = match.group(1).strip() # Get matched group and strip whitespace
                self.logger.debug("Found JSON block in Gemini response.")
            else:
                self.logger.warning("JSON markdown block not found. Attempting to parse entire response.")
                json_str = instructions_text.strip()

            instructions_list = json.loads(json_str)
            if not isinstance(instructions_list, list):
                 self.logger.error("Parsed Gemini JSON is not a list.")
                 return None

            valid_instructions = []
            for item in instructions_list:
                 if isinstance(item, dict) and "action" in item:
                      valid_instructions.append({"action": item["action"], "target": item.get("target")})
                 else:
                      self.logger.warning("Skipping invalid instruction item from Gemini: %s", item)
            self.logger.info("Successfully parsed %d valid instructions from Gemini.", len(valid_instructions))
            return valid_instructions

        except json.JSONDecodeError as e:
            self.logger.error("Failed to parse Gemini response as JSON: %s", e)
            self.logger.debug("Failed JSON content: %s", json_str) # Log content that failed
            return None
        except Exception as e:
             self.logger.error("Unexpected error parsing Gemini instructions: %s", e, exc_info=True)
             return None

    def map_actions_to_robot_commands(self, instructions):
        # (Keep implementation structure, add logging)
        robot_commands = []
        if instructions is None: return None
        self.logger.info("Mapping %d high-level instructions to robot commands...", len(instructions))
        for i, instruction in enumerate(instructions):
            action = instruction.get("action")
            target = instruction.get("target")
            self.logger.debug("Mapping instruction %d: action=%s, target=%s", i+1, action, target)

            # TODO: Implement robust mapping & call path planner here
            if action == "error":
                self.logger.error("Gemini reported planning error: %s", instruction.get('message', 'Unknown error'))
                continue
            elif action == "go_to":
                if target is None:
                    self.logger.warning("Skipping 'go_to' action: missing target.")
                    continue
                # --- Integrate Path Planner Call ---
                # waypoints = planner.plan_path(self.robot_state, target) # Assuming planner exists
                # if waypoints:
                #     for wp in waypoints: robot_commands.append(("move_to", wp))
                # else:
                #     self.logger.error("Path planning failed for target: %s", target)
                # --- End Integration Placeholder ---
                robot_commands.append(("move_to", target)) # Placeholder command
            elif action == "scan_area":
                 robot_commands.append(("scan", target))
            elif action == "approach":
                 robot_commands.append(("approach", target))
            elif action == "report_status":
                 robot_commands.append(("report_status", None))
            else:
                 self.logger.warning("Unrecognized action '%s' encountered during mapping.", action)

        self.logger.info("Generated %d low-level robot commands.", len(robot_commands))
        return robot_commands

    def execute_commands(self, commands):
        # (Keep implementation structure, add logging)
        if not commands:
            self.logger.info("No commands to execute.")
            return
        self.logger.info("Executing %d robot commands...", len(commands))
        for i, (command, args) in enumerate(commands):
             self.logger.info("CMD %d: Executing %s with args: %s", i+1, command, args)
             # TODO: Add actual robot control interface calls within try-except blocks
             try:
                 if command == "move_to":
                      # send_move_command(args) # Example
                      time.sleep(0.5) # Simulate execution
                 elif command == "scan":
                      # send_scan_command(args)
                      time.sleep(0.3)
                 elif command == "approach":
                      # send_approach_command(args)
                      time.sleep(0.4)
                 elif command == "report_status":
                      # report_status()
                      time.sleep(0.1)
                 else:
                      self.logger.warning("Skipping unrecognized command '%s' during execution.", command)
             except Exception as e:
                  self.logger.error("Error executing command %s(%s): %s", command, args, e, exc_info=True)
                  # Decide: Stop execution? Continue with next command?
                  # For now, continue:
                  # break # <-- Uncomment to stop on first execution error

        self.logger.info("Finished executing commands.")


    def run_navigation(self, user_command):
        # (Keep implementation from previous version, relies on logging within called methods)
        instructions_text = self.get_gemini_navigation_instructions(user_command)
        if instructions_text:
            instructions = self.parse_gemini_instructions(instructions_text)
            if instructions:
                if instructions[0].get("action") == "error":
                     self.logger.error("Gemini could not generate plan: %s", instructions[0].get('message'))
                     return
                commands = self.map_actions_to_robot_commands(instructions)
                if commands:
                    self.execute_commands(commands)
                else:
                    self.logger.error("Action mapping failed. No commands generated.")
            else:
                self.logger.error("Could not parse Gemini instructions.")
        else:
            self.logger.error("Could not get navigation instructions from Gemini.")


# --- Path Planner (RRT*) ---
# (Keep PathPlannerRRTStar class definition from previous version)
# Remember to add logging inside its methods: self.logger = logging.getLogger(__name__)
class PathPlannerRRTStar:
    # ... (Full implementation from previous answer) ...
    # Add self.logger = logging.getLogger(__name__) in __init__
    # Replace print() with self.logger.info/warning/error/debug
    # Add try-except around critical sections if needed (e.g., KDTree operations)
    def __init__(self, behavior_learner, config_space_bounds, step_size=DEFAULT_STEP_SIZE, max_iter=DEFAULT_MAX_ITERATIONS, goal_sample_rate=DEFAULT_GOAL_SAMPLE_RATE, search_radius=DEFAULT_SEARCH_RADIUS, robot_radius=ROBOT_RADIUS):
        self.logger = logging.getLogger(__name__)
        self.behavior_learner = behavior_learner
        # ... rest of __init__ ...
        self.logger.info(f"RRT* Planner Initialized. Using KDTree: {USE_KDTREE}")

    def update_static_obstacles(self, obstacles_list):
        self.static_obstacles = obstacles_list
        self.logger.info("Updated static obstacles: %d obstacles loaded.", len(self.static_obstacles))

    def predict_dynamic_obstacles(self, robot_pose, prediction_horizon_secs, time_step):
         # TODO: Implement prediction logic using self.behavior_learner
         self.logger.debug("Predicting dynamic obstacles...")
         # ... implementation ...
         return {} # Placeholder

    def get_goal_position(self, target, current_graph):
         # ... implementation ...
         if isinstance(target, str):
              # ... lookup logic ...
              if state:
                   self.logger.info("Target '%s' resolved to graph coords: %s", target, tuple(state[:2]))
                   return tuple(state[:2])
         elif isinstance(target, (list, tuple, np.ndarray)):
              return tuple(target[:2])
         self.logger.warning("Could not resolve target '%s' to coordinates.", target)
         return None

    def _is_collision(self, node1, node2, dynamic_obstacles_pred):
        # ... (Keep implementation, add logging for detected collisions if debugging) ...
        # Example: if collision_detected: self.logger.debug("Collision between %s and %s", node1, node2); return True
        try:
            # ... (collision check logic) ...
             pass # Keep existing logic for now
        except Exception as e:
             self.logger.error("Error during collision check between %s and %s: %s", node1, node2, e, exc_info=True)
             return True # Assume collision on error
        return False # Placeholder return

    def plan_path(self, start_pose, goal_target):
        start_time = time.time()
        self.logger.info("Starting RRT* planning from %s to target '%s'", start_pose[:2], goal_target)
        # ... (Rest of plan_path implementation from previous answer) ...
        # Replace print statements with self.logger calls

        # Example modification for logging results:
        elapsed_time = time.time() - start_time
        if best_goal_node:
            path = self._reconstruct_path(best_goal_node)
            self.logger.info("RRT* SUCCESS: Path found (%d waypoints, cost %.2f) in %.2f seconds.", len(path), best_goal_node.cost, elapsed_time)
            return path
        else:
            # ... (logic to find closest node) ...
            if closest_node_to_goal:
                 path = self._reconstruct_path(closest_node_to_goal)
                 self.logger.warning("RRT* PARTIAL: Direct goal not reached. Returning path to closest node (%d waypoints) after %.2f seconds.", len(path), elapsed_time)
                 return path
            else:
                 self.logger.error("RRT* FAILED: No path found after %d iterations in %.2f seconds.", self.max_iter, elapsed_time)
                 return None


# --- Placeholder Functions ---
# Add logging within these as they are implemented
def get_user_command():
    logger.debug("Checking for user command...")
    return None

def get_robot_state():
    # logger.debug("Getting robot state...")
    return np.array([0.0, 0.0, 0.0, 0.0]) # Example [x, y, vx, vy]

def get_environment_info_for_combiner():
    global environment_info_global
    # logger.debug("Getting environment info for combiner...")
    if environment_info_global is not None:
        return environment_info_global.astype(np.float32)
    else:
        return np.zeros(2, dtype=np.float32) # Match example global shape

def process_lidar_data_for_obstacles(lidar_data):
    # logger.debug("Processing lidar for obstacles...")
    if lidar_data is None: return np.zeros(1, dtype=np.float32)
    return np.array([np.mean(lidar_data)], dtype=np.float32)

def smooth_path(waypoints):
    logger.info("Smoothing path (Placeholder)...")
    return waypoints

def execute_path(path):
    logger.info("Executing path (Placeholder)...")
    if path is None: return
    for i, waypoint in enumerate(path):
        logger.debug("  -> Moving towards waypoint %d: %s", i+1, waypoint)
        time.sleep(0.1) # Simulate movement

def detect_objects(frame, detector):
     logger.debug("Detecting objects...")
     # TODO: Implement real detection + Add try-except
     try:
        # Dummy detections:
        detections = [
             {'id': f'obj_{int(time.time()) % 100 + i}', 'measurement': np.array([random.uniform(1,19), random.uniform(1,14)]), 'confidence': 0.8+random.random()*0.1, 'class': random.choice(['cup', 'book', 'obstacle'])} for i in range(random.randint(1,3))
        ]
        return detections
     except Exception as e:
         logger.error("Error during object detection: %s", e, exc_info=True)
         return []

# --- Gemini Object Query Placeholders (Add logging) ---
def object_is_unexplored(object_id, graph):
    # logger.debug("Checking if %s is unexplored", object_id)
    return not graph.nodes.get(object_id, {}).get('explored', False)
def object_confidence_is_low(object_id, graph):
    # logger.debug("Checking confidence for %s", object_id)
    return graph.nodes.get(object_id, {}).get('gemini_confidence', 0.0) < 0.7
def capture_object_image(frame, detection):
    logger.debug("Capturing image for %s", detection.get('id'))
    return frame[0:50, 0:50]
def query_gemini_for_object(image, question):
    logger.info("Querying Gemini about object image: '%s'", question)
    return {'label': 'cup', 'description': 'A white ceramic cup.', 'confidence': 0.95} # Dummy
def update_object_graph_with_gemini(graph, object_id, gemini_response):
    logger.info("Updating graph for %s with Gemini info", object_id)
    # ... (update logic) ...
def should_explore_object(object_id, graph):
    # logger.debug("Deciding exploration for %s", object_id)
    return False
def explore_object(object_id, navigator):
    logger.info("Initiating exploration for %s", object_id)
    # command = f"Approach object {object_id} and scan it."
    # navigator.run_navigation(command)


# --- Main Robot Loop ---
def robot_loop():
    loop_logger = logging.getLogger("RobotLoop")
    loop_logger.info("--- Starting Robot Loop ---")

    # Initialization Block
    try:
        loop_logger.info("Initializing models and components...")
        behavior_learner = ObjectBehaviorLearner(state_dimension=KALMAN_STATE_DIM)

        # Calculate combiner dims (Example - needs refinement based on actual feature sizes)
        env_dim = 2 # From example global Lidar info
        robot_dim = KALMAN_STATE_DIM
        obstacle_dim = 1 # From example obstacle processing
        context_dim = env_dim + robot_dim + obstacle_dim
        transformer_output_dim = KALMAN_STATE_DIM
        combiner_output_dim = 2 # Example: Predict [vx, vy] or [x,y] waypoints
        prediction_horizon = 10

        combiner = PredictionCombiner(
            transformer_output_dim=transformer_output_dim,
            context_dim=context_dim,
            output_dim=combiner_output_dim,
            prediction_horizon=prediction_horizon
        )
        combiner.compile(optimizer='adam', loss='mse') # TODO: Choose appropriate loss

        navigator = RobotNavigator(object_behavior_learner=behavior_learner)

        # Initialize Planner
        planner = PathPlannerRRTStar(behavior_learner, CONFIG_SPACE_BOUNDS,
                                   step_size=STEP_SIZE, max_iter=MAX_ITERATIONS,
                                   goal_sample_rate=GOAL_SAMPLE_RATE,
                                   search_radius=SEARCH_RADIUS, robot_radius=ROBOT_RADIUS)
        # TODO: Load or define static obstacles for the planner
        static_obstacles_example = [
             Obstacle(x=5, y=5, radius=1.0), Obstacle(x=10, y=8, radius=1.5), Obstacle(x=15, y=10, radius=1.0)
        ]
        planner.update_static_obstacles(static_obstacles_example)


        trackers = {}
        object_state_history = {} # {object_id: deque(maxlen=history_length)}
        history_length = 10

        loop_logger.info("Initializing ROS2 node...")
        rclpy.init(args=None)
        environment_info_node = EnvironmentInfoNode()

        loop_logger.info("Opening video capture...")
        capture = cv2.VideoCapture(0) # TODO: Check index
        if not capture.isOpened():
            raise IOError("Cannot open webcam")

        robot_id = 'robot_1'
        frame_count = 0
        loop_logger.info("Initialization complete.")

    except Exception as e:
        loop_logger.critical("Initialization failed: %s. Exiting.", e, exc_info=True)
        # Attempt cleanup even if init failed partially
        if 'capture' in locals() and isinstance(capture, cv2.VideoCapture) and capture.isOpened(): capture.release()
        if 'rclpy' in sys.modules and rclpy.ok(): rclpy.shutdown()
        sys.exit(1)

    # Main Loop Block
    try:
        while rclpy.ok():
            # --- ROS2 ---
            rclpy.spin_once(environment_info_node, timeout_sec=0.01)

            # --- Perception ---
            ret, frame = capture.read()
            if not ret:
                loop_logger.warning("Failed to read frame from camera. Skipping iteration.")
                time.sleep(0.1)
                continue
            frame_count += 1
            if frame_count % 100 == 0: loop_logger.debug("Processing frame %d", frame_count)

            # --- Main Processing Block (Wrapped for Iteration Robustness) ---
            try:
                detections = detect_objects(frame, object_detector)
                current_robot_state = get_robot_state()
                current_environment_info_comb = get_environment_info_for_combiner()
                current_obstacles_comb = process_lidar_data_for_obstacles(environment_info_global)

                processed_object_ids = set()
                object_predictions_transformer = {}
                object_actual_states = {}

                # --- Tracking & Prediction Loop ---
                for detection in detections:
                     # Inner try-except to handle errors per-detection
                     try:
                        object_id = detection.get('id')
                        measurement = detection.get('measurement')
                        if object_id is None or measurement is None: continue
                        processed_object_ids.add(object_id)

                        # Kalman Init/Update
                        if object_id not in trackers:
                            initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
                            trackers[object_id] = KalmanTracker(initial_state)
                            object_state_history[object_id] = deque(maxlen=history_length) # Use deque
                        predicted_state_kf = trackers[object_id].predict() # Kalman prediction
                        updated_state = trackers[object_id].update(measurement) # Kalman update

                        if updated_state is None: # Kalman update failed
                            loop_logger.warning("Kalman update failed for %s", object_id)
                            continue # Skip processing this object further

                        current_state_flat = updated_state.flatten()
                        object_actual_states[object_id] = current_state_flat

                        # State History
                        object_state_history[object_id].append(current_state_flat)

                        # Transformer Prediction
                        if len(object_state_history[object_id]) >= history_length:
                             history_array = np.array(list(object_state_history[object_id])).reshape(1, history_length, KALMAN_STATE_DIM)
                             transformer_pred_seq = behavior_learner.predict(history_array)
                             if transformer_pred_seq is not None:
                                 current_transformer_pred = transformer_pred_seq[0, -1, :]
                                 object_predictions_transformer[object_id] = current_transformer_pred
                                 # Update graph using Transformer prediction
                                 behavior_learner.update_graph(object_id, updated_state, current_transformer_pred)
                             else:
                                 # Handle prediction failure, maybe use KF prediction for graph
                                 if predicted_state_kf is not None:
                                      behavior_learner.update_graph(object_id, updated_state, predicted_state_kf)
                        else:
                             # Not enough history, use KF prediction for graph update
                             if predicted_state_kf is not None:
                                 behavior_learner.update_graph(object_id, updated_state, predicted_state_kf)

                        # --- Optional Gemini Object Query ---
                        # Wrap this section in its own try-except if implemented
                        # if object_is_unexplored(...) or object_confidence_is_low(...):
                        #    try:
                        #       # ... query logic ...
                        #    except Exception as e_gemini_obj:
                        #       loop_logger.error(...)


                     except ValueError as e_val: # Catch specific errors like Kalman init failure
                          loop_logger.error("ValueError processing detection for %s: %s", detection.get('id', 'UNKNOWN'), e_val, exc_info=True)
                     except Exception as e_det: # Catch other unexpected errors for this detection
                          loop_logger.error("Unhandled error processing detection %s: %s", detection.get('id', 'UNKNOWN'), e_det, exc_info=True)

                # --- Remove Old Trackers ---
                lost_track_ids = set(trackers.keys()) - processed_object_ids
                for lost_id in lost_track_ids:
                    loop_logger.info("Object %s lost track.", lost_id)
                    if lost_id in trackers: del trackers[lost_id]
                    if lost_id in object_state_history: del object_state_history[lost_id]
                    # No need to delete from object_predictions_transformer (rebuilt each frame)

                # --- Combiner Prediction ---
                waypoints = None
                if object_predictions_transformer: # Check if we have *any* transformer predictions
                    try:
                        # --- Prepare inputs for combiner (needs careful thought) ---
                        # Example: Using average state of *all* tracked objects as input
                        # THIS IS LIKELY TOO SIMPLISTIC - Need better feature engineering
                        if object_actual_states:
                             avg_actual_state = np.mean(np.array(list(object_actual_states.values())), axis=0)
                             avg_transformer_pred = np.mean(np.array(list(object_predictions_transformer.values())), axis=0) # Use corresponding predictions

                             errors = [np.linalg.norm(object_actual_states[oid] - object_predictions_transformer[oid])
                                       for oid in object_predictions_transformer if oid in object_actual_states]
                             avg_error = np.mean(errors) if errors else 1.0
                             transformer_confidence = np.array([1.0 / (1.0 + avg_error)], dtype=np.float32) # Crude confidence

                             combiner_input_list = [
                                 tf.constant(avg_transformer_pred.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(transformer_confidence.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_environment_info_comb.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_robot_state.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_obstacles_comb.reshape(1, -1), dtype=tf.float32)
                             ]
                             waypoints_tensor = combiner(combiner_input_list, training=False)
                             if waypoints_tensor is not None:
                                 waypoints = waypoints_tensor.numpy()[0]
                                 # loop_logger.debug("Combiner generated waypoints.")
                        else:
                            loop_logger.warning("No actual states available to average for combiner input.")

                    except tf.errors.InvalidArgumentError as e:
                        loop_logger.error("TF InvalidArgumentError during combiner prediction: %s", e, exc_info=True)
                    except Exception as e:
                        loop_logger.error("Error during prediction combination: %s", e, exc_info=True)

                # --- Navigation Decision ---
                user_command = get_user_command()

                if user_command:
                    loop_logger.info("Processing user command: '%s'", user_command)
                    try:
                        navigator.update_context(current_robot_state, current_environment_info_comb) # Use consistent env info?
                        navigator.run_navigation(user_command)
                    except Exception as e:
                        loop_logger.error("Error during command execution flow: %s", e, exc_info=True)
                elif waypoints is not None:
                    try:
                        # loop_logger.debug("Following combiner waypoints.")
                        smoothed_path = smooth_path(waypoints)
                        execute_path(smoothed_path)
                    except Exception as e:
                        loop_logger.error("Error executing combiner-generated path: %s", e, exc_info=True)
                else:
                    loop_logger.debug("Idle: No command or waypoints.")
                    time.sleep(0.1) # Prevent busy-waiting when idle

                # --- Periodic Graph Saving ---
                if frame_count % 300 == 0: # Less frequent saving?
                    loop_logger.info("Saving graph at frame %d...", frame_count)
                    save_graph_to_cloud_storage(behavior_learner.object_graph, GCS_BUCKET_NAME, GCS_BLOB_NAME)

            except Exception as e:
                 loop_logger.error("Unhandled error in main loop iteration: %s", e, exc_info=True)
                 time.sleep(1) # Pause briefly after an error

            # --- Optional: Add visualization ---
            # cv2.imshow('Frame', frame)
            # key = cv2.waitKey(1) & 0xFF
            # if key == ord('q'):
            #      loop_logger.info("Quit signal received via OpenCV window.")
            #      break

    except KeyboardInterrupt:
        loop_logger.info("Loop interrupted by user (Ctrl+C).")
    except Exception as e:
        loop_logger.critical("Critical unhandled error in robot_loop: %s", e, exc_info=True)
    finally:
        # --- Cleanup ---
        loop_logger.info("Shutting down...")
        if 'capture' in locals() and isinstance(capture, cv2.VideoCapture) and capture.isOpened(): capture.release()
        cv2.destroyAllWindows()
        if 'rclpy' in sys.modules and rclpy.ok():
            if 'environment_info_node' in locals() and isinstance(environment_info_node, Node):
                 loop_logger.info("Destroying ROS2 node...")
                 environment_info_node.destroy_node()
            loop_logger.info("Shutting down ROS2...")
            rclpy.shutdown()
        loop_logger.info("Cleanup complete.")


# --- Main Execution Guard ---
if __name__ == '__main__':
    logger.info("========================================")
    logger.info("   Starting Robot Application           ")
    logger.info("========================================")
    try:
        robot_loop()
    except SystemExit as e:
         logger.warning("Application exited with code %s", e.code)
    except Exception as e:
        logger.critical("Application crashed at top level: %s", e, exc_info=True)
    finally:
        logger.info("========================================")
        logger.info("   Robot Application Finished           ")
        logger.info("========================================")
        logging.shutdown() # Flush and close logging handlers
