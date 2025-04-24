# Main Robot Application Code
# Features:
# - TensorFlow-based architecture
# - Placeholder for Neptune Graph Database interaction
# - Placeholder for Federated Learning client
# - RRT* Path Planner structure
# - BNN Uncertainty Consultant module (using MC Dropout concept)
# - Enhanced Logging and Error Handling

# --- Imports ---
import os
import json
import networkx as nx
import google.generativeai
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, TimeDistributed,
                                     MultiHeadAttention, LayerNormalization, Dropout) # Removed unused layers like LSTM, Attention, etc. for clarity
from tensorflow.keras.models import Model
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import google.cloud.pubsub_v1 as pubsub
from datetime import datetime, timezone, timedelta # Use timezone
from google.cloud import storage
import logging
import time
import sys
import math
import random
import re
from collections import namedtuple, deque, OrderedDict

# Optional KDTree for RRT*
try:
    from scipy.spatial import KDTree
    USE_KDTREE = True
except ImportError:
    USE_KDTREE = False
    # Logging not set up yet, use print for this initial warning
    print("Warning: scipy.spatial.KDTree not found. Falling back to slower nearest neighbor search.")

# Optional TFF for Federated Learning Placeholder
try:
    import tensorflow_federated as tff
    TFF_AVAILABLE = True
except ImportError:
    TFF_AVAILABLE = False
    print("Warning: tensorflow_federated not installed. FL client simulation will be limited.")


# Google Cloud Exceptions
from google.api_core import exceptions as google_api_exceptions
from google.cloud import exceptions as google_cloud_exceptions

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
log_level = logging.INFO # Default level
# Set level to DEBUG for more verbose development output
# log_level = logging.DEBUG

root_logger = logging.getLogger()
# Clear existing handlers if any (useful in interactive environments)
# if root_logger.hasHandlers():
#     root_logger.handlers.clear()

if not root_logger.hasHandlers(): # Setup handlers only once
    # Console Handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    # File Handler (using RotatingFileHandler)
    log_file = 'robot_app.log'
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3) # 5MB * 3 backups
        file_handler.setFormatter(log_formatter)
    except ImportError:
        print(f"Warning: RotatingFileHandler not available. Using basic FileHandler for {log_file}.")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)

    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # General logger for top-level scope

# --- Configuration ---
logger.info("Loading configuration...")
# TODO: Use a dedicated config file/system (e.g., YAML, dotenv, Python config file)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
YOLO_CFG_PATH = os.environ.get("YOLO_CFG_PATH", "yolov3.cfg")
YOLO_WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", "yolov3.weights")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", 'your-gcp-project-id')
PUBSUB_TOPIC_ID = 'object_exploration'
PUBSUB_SUBSCRIPTION_ID = 'robot_subscription'
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'your-bucket-name')
NEPTUNE_API_ENDPOINT = os.environ.get("NEPTUNE_API_ENDPOINT", "http://placeholder.neptune.api/graph") # Placeholder
FL_SERVER_URL = os.environ.get("FL_SERVER_URL", "http://placeholder.fl.server") # Placeholder

# RRT* Planner Configuration
CONFIG_SPACE_BOUNDS = (0, 20, 0, 15) # Example: (min_x, max_x, min_y, max_y)
STEP_SIZE = 0.5
MAX_ITERATIONS = 5000 # Increase for denser sampling if needed
GOAL_SAMPLE_RATE = 0.1 # Bias towards goal
SEARCH_RADIUS = 1.5   # RRT* rewiring radius
ROBOT_RADIUS = 0.3    # For collision checking
OBJECT_BASE_RADIUS = 0.2 # Assumed base radius for dynamic obstacles
PLANNER_SAFETY_FACTOR = 1.5 # Scales uncertainty std dev to safety margin

# Uncertainty Consultant Configuration
UNCERTAINTY_UPDATE_HZ = 1.0 # How often to run BNN analysis
BNN_MC_SAMPLES = 30       # Number of samples for MC Dropout
HIGH_UNCERTAINTY_THRESHOLD = 0.5 # Variance threshold example

# --- Configuration Validation ---
# Basic checks - enhance as needed
if GEMINI_API_KEY == "YOUR_API_KEY_HERE": logger.warning("GEMINI_API_KEY not set.")
if GCP_PROJECT_ID == 'your-gcp-project-id': logger.warning("GCP_PROJECT_ID not set.")
if GCS_BUCKET_NAME == 'your-bucket-name': logger.warning("GCS_BUCKET_NAME not set.")
if not os.path.exists(YOLO_CFG_PATH): logger.error("YOLO Cfg not found: %s", YOLO_CFG_PATH)
if not os.path.exists(YOLO_WEIGHTS_PATH): logger.error("YOLO Weights not found: %s", YOLO_WEIGHTS_PATH)
# Add checks for other essential configs

# --- Configure Gemini API ---
gemini_configured = False
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
        google.generativeai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured.")
        gemini_configured = True
    else: logger.error("Skipping Gemini API configuration: Missing/placeholder key.")
except Exception as e: logger.critical("Failed to configure Gemini API: %s", e, exc_info=True)

# --- Object Detection ---
logger.info("Loading Object Detector (YOLO)...")
object_detector = None
try:
    if not os.path.exists(YOLO_CFG_PATH): raise FileNotFoundError(f"YOLO Cfg not found: {YOLO_CFG_PATH}")
    if not os.path.exists(YOLO_WEIGHTS_PATH): raise FileNotFoundError(f"YOLO Weights not found: {YOLO_WEIGHTS_PATH}")
    object_detector = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    logger.info("Object Detector loaded successfully.")
except (cv2.error, FileNotFoundError) as e:
    logger.critical("Failed to load YOLO model: %s", e, exc_info=True)
    sys.exit(1) # Assuming detector is essential

# ==============================================================================
# === Class Definitions ========================================================
# ==============================================================================

# --- Kalman Filter ---
KALMAN_STATE_DIM = 4
KALMAN_MEASURE_DIM = 2
class KalmanTracker:
    # (Implementation from previous step - including __init__, predict, update)
    def __init__(self, initial_state):
        self.logger = logging.getLogger(__name__)
        if initial_state is None or initial_state.shape[0] != KALMAN_STATE_DIM:
             msg = f"Initial state must have dimension {KALMAN_STATE_DIM}, got shape {initial_state.shape if initial_state is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)
        # IMPORTANT: Matrices below NEED TUNING
        self.filter = cv2.KalmanFilter(KALMAN_STATE_DIM, KALMAN_MEASURE_DIM)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        dt = 1.0 # MUST replace with actual time delta
        self.filter.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.filter.processNoiseCov = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.03 # Tune
        self.filter.measurementNoiseCov = np.eye(KALMAN_MEASURE_DIM, dtype=np.float32) * 0.1 # Tune
        self.filter.statePost = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.statePre = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.errorCovPost = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune
        self.filter.errorCovPre = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune

    def predict(self):
        try:
            predicted_state = self.filter.predict()
            return predicted_state
        except cv2.error as e:
            self.logger.error("OpenCV Kalman predict error: %s", e, exc_info=True)
            return None

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
            return self.filter.statePost # Return previous state


# --- Transformer Model (TensorFlow) ---
class TransformerBlock(tf.keras.layers.Layer):
    # (Implementation from previous step)
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads; self.dff = dff; self.rate = rate
        # Calculate key_dim carefully
        key_dim_calc = self.d_model // self.num_heads if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads == 0 else self.d_model
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim_calc, name=f"{kwargs.get('name', 'transformer_block')}_mha")
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f"{kwargs.get('name', 'transformer_block')}_ln1")
        self.ffn = tf.keras.Sequential(
            [Dense(self.dff, activation='relu'), Dense(self.d_model)],
            name=f"{kwargs.get('name', 'transformer_block')}_ffn"
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f"{kwargs.get('name', 'transformer_block')}_ln2")
        self.dropout1 = Dropout(self.rate, name=f"{kwargs.get('name', 'transformer_block')}_drop1")
        self.dropout2 = Dropout(self.rate, name=f"{kwargs.get('name', 'transformer_block')}_drop2")

    def call(self, x, training=None, mask=None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training) # Removed return_attention_scores for simplicity
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class ObjectBehaviorLearner:
    # (Implementation from previous step - builds model using TransformerBlock)
    def __init__(self, prediction_horizon=10, state_dimension=KALMAN_STATE_DIM, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(f"{self.__class__.__name__}") # Class specific logger
        self.prediction_horizon = prediction_horizon # Note: Model predicts output for input steps, not future horizon currently
        self.state_dimension = state_dimension
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.local_object_graph_cache = nx.Graph()
        self.logger.info("Initializing ObjectBehaviorLearner (local graph acts as cache).")

        try:
            if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads != 0:
                self.logger.warning("d_model (%d) not divisible by num_heads (%d).", self.d_model, self.num_heads)

            # Define layers reused in the inner model
            self.embedding_layer = Dense(d_model, name="input_embedding")
            self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff, rate=0.1, name=f"transformer_block_{i}") for i in range(num_layers)] # Standard dropout rate
            self.output_dense_layer = TimeDistributed(Dense(self.state_dimension), name="output_dense")

            # Define the inner Keras Model using subclassing
            class TransformerModel(tf.keras.Model):
                def __init__(self, embedding_layer, transformer_blocks, output_dense_layer, **kwargs):
                    super().__init__(**kwargs)
                    self.embedding_layer = embedding_layer
                    self.transformer_blocks = transformer_blocks
                    self.output_dense_layer = output_dense_layer
                    # Positional encoding could be added here as a layer if needed

                @tf.function # Apply tf.function for potential graph optimization
                def call(self, inputs, training=None, mask=None):
                    x = self.embedding_layer(inputs)
                    # Add positional encoding here: x += self.pos_encoding(...)
                    for block in self.transformer_blocks:
                        x = block(x, training=training, mask=mask)
                    return self.output_dense_layer(x)

            self.model = TransformerModel(
                self.embedding_layer, self.transformer_blocks, self.output_dense_layer
            )
            # Build model
            dummy_input_shape = (None, None, state_dimension) # Use None for flexible batch/time steps
            self.model.build(input_shape=dummy_input_shape)
            self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
            self.logger.info("ObjectBehaviorLearner TF Model built and compiled.")
            # self.model.summary(print_fn=self.logger.info)

        except Exception as e:
            self.logger.critical("Failed to build/compile ObjectBehaviorLearner model: %s", e, exc_info=True)
            raise

    def train(self, past_states, target_states):
        self.logger.info("Starting training...")
        try:
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            val_loss = history.history.get('val_loss', [None])[-1]
            self.logger.info("Training finished. Final validation loss: %s", f"{val_loss:.4f}" if val_loss else "N/A")
        except Exception as e:
            self.logger.error("Error during training: %s", e, exc_info=True)

    def predict(self, past_states):
        """Standard point estimate prediction."""
        # self.logger.debug("Standard prediction...") # Can be verbose
        try:
            # Use call directly with training=False for consistency with BNN version
            predictions = self.model(past_states, training=False)
            return predictions
        except Exception as e:
             self.logger.error("Error during standard prediction: %s", e, exc_info=True)
             return None

    def update_local_cache(self, object_id, state, timestamp_iso):
        state_values = state.flatten().tolist()
        if object_id not in self.local_object_graph_cache.nodes:
            self.local_object_graph_cache.add_node(object_id, state=state_values, timestamp=timestamp_iso)
            # self.logger.debug("Added node %s to local cache", object_id)
        else:
            self.local_object_graph_cache.nodes[object_id]['state'] = state_values
            self.local_object_graph_cache.nodes[object_id]['timestamp'] = timestamp_iso
            # self.logger.debug("Updated node %s in local cache", object_id)


# --- BNN Variant using MC Dropout ---
class BNNObjectBehaviorLearner(ObjectBehaviorLearner):
    """BNN version using MC Dropout for uncertainty."""
    def __init__(self, num_mc_samples=BNN_MC_SAMPLES, **kwargs):
        super().__init__(**kwargs)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not isinstance(num_mc_samples, int) or num_mc_samples <= 1:
            msg = "num_mc_samples must be an integer > 1."
            self.logger.error(msg)
            raise ValueError(msg)
        self.num_mc_samples = num_mc_samples
        self.logger.info("Initialized with %d MC samples.", self.num_mc_samples)
        # Verify dropout layers exist
        has_dropout = any(isinstance(layer, Dropout) for layer in self._get_all_layers(self.model))
        if not has_dropout:
             self.logger.warning("MC Dropout requires Dropout layers, but none were detected. Ensure model config includes them.")

    def _get_all_layers(self, model):
        """Helper to recursively get all layers including nested ones."""
        layers = []
        for layer in model.layers:
             layers.append(layer)
             if hasattr(layer, 'layers'): # Check if it's a nested model/layer
                 layers.extend(self._get_all_layers(layer))
        return layers

    @tf.function # Decorate with tf.function for potential performance boost
    def predict_mc_dropout_tf(self, past_states):
        """Internal TF function for the prediction loop."""
        # Use tf.TensorArray for potentially better performance with tf.function
        predictions_array = tf.TensorArray(dtype=tf.float32, size=self.num_mc_samples, dynamic_size=False)
        for i in tf.range(self.num_mc_samples):
            predictions = self.model(past_states, training=True) # training=True is key
            predictions_array = predictions_array.write(i, predictions)
        return predictions_array.stack() # Stack results

    def predict(self, past_states: np.ndarray | tf.Tensor) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """ Performs prediction using MC Dropout to estimate mean and variance. """
        start_time = time.time()
        # self.logger.debug("BNN MC Dropout prediction started...") # Can be verbose
        try:
            if not isinstance(past_states, tf.Tensor):
                past_states = tf.constant(past_states, dtype=tf.float32)

            predictions_stack = self.predict_mc_dropout_tf(past_states)
            # Shape: (num_mc_samples, batch_size, time_steps, state_dimension)

            mean_prediction = tf.reduce_mean(predictions_stack, axis=0)
            variance_prediction = tf.math.reduce_variance(predictions_stack, axis=0)

            duration = time.time() - start_time
            # self.logger.debug("MC Dropout prediction finished in %.3f s.", duration)
            return mean_prediction, variance_prediction

        except Exception as e:
            self.logger.error("Error during MC Dropout prediction: %s", e, exc_info=True)
            return None, None


# --- Prediction Combiner (TensorFlow) ---
class PredictionCombiner(tf.keras.Model):
    # (Implementation from previous step - unchanged)
    def __init__(self, transformer_output_dim, context_dim, output_dim=KALMAN_STATE_DIM, prediction_horizon=10, use_confidence_weighting=True, dropout_rate=0.2):
        super(PredictionCombiner, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.use_confidence_weighting = use_confidence_weighting
        self.dropout_rate = dropout_rate
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim
        # Define layers
        self.dense1 = Dense(128, activation='relu', name="combiner_dense1")
        self.dense2 = Dense(64, activation='relu', name="combiner_dense2")
        self.dense3 = Dense(self.output_dim * self.prediction_horizon, name="combiner_output_dense")
        self.layer_norm = LayerNormalization(epsilon=1e-6, name="combiner_layernorm")
        self.dropout = Dropout(self.dropout_rate, name="combiner_dropout")
        self.logger.info("Prediction Combiner Initialized.")
        # TODO: Build the model here with expected input shapes for better error checking/summary
        # self.build(input_shape=[(None, transformer_output_dim), (None, 1), ...]) # Example shape list

    def call(self, inputs, training=None):
        try:
            # TODO: Improve input handling robustness (check list length, tensor types/shapes)
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
                 self.logger.error("PredictionCombiner received invalid inputs structure.")
                 return None
            transformer_predictions, transformer_confidences, environment_info, robot_state, obstacles = inputs

            # Flattening logic using tf.reshape for better control
            def safe_flatten(tensor):
                shape = tf.shape(tensor)
                if len(shape) > 2:
                    return tf.reshape(tensor, [shape[0], -1]) # Reshape to (batch_size, flattened_features)
                elif len(shape) == 1: # Handle single dimension input (e.g., confidence)
                     return tf.expand_dims(tensor, axis=-1) # Ensure shape (batch_size, 1)
                return tensor # Assume shape is already (batch_size, features) or similar

            transformer_predictions_flat = safe_flatten(transformer_predictions)
            transformer_confidences_flat = safe_flatten(transformer_confidences)
            environment_info_flat = safe_flatten(environment_info)
            robot_state_flat = safe_flatten(robot_state)
            obstacles_flat = safe_flatten(obstacles)

            # Confidence Weighting
            weighted_preds = transformer_predictions_flat
            if self.use_confidence_weighting and transformer_confidences_flat is not None:
                 # Make sure confidence shape is broadcastable, e.g., (batch, 1)
                 confidences_exp = transformer_confidences_flat
                 if len(tf.shape(confidences_exp)) == 1: confidences_exp = tf.expand_dims(confidences_exp, axis=-1)
                 weighted_preds = transformer_predictions_flat * confidences_exp # Element-wise multiplication

            # Concatenate features
            combined_input = Concatenate()([weighted_preds, environment_info_flat, robot_state_flat, obstacles_flat])

            x = self.layer_norm(combined_input, training=training)
            x = self.dropout(x, training=training)
            x = self.dense1(x)
            x = self.dense2(x)
            waypoints_flat = self.dense3(x)

            # Reshape into waypoints: (batch_size, prediction_horizon, output_dim)
            final_shape = [-1, self.prediction_horizon, self.output_dim] # Use -1 for batch size
            waypoints = tf.reshape(waypoints_flat, final_shape)
            return waypoints

        except (tf.errors.InvalidArgumentError, ValueError) as e: # Catch specific TF/shape errors
             self.logger.error("TF Error in Combiner call (check input shapes/types/concat): %s", e, exc_info=True)
             return None
        except Exception as e:
             self.logger.error("Unexpected error in PredictionCombiner call: %s", e, exc_info=True)
             return None


# --- ROS2 Node ---
environment_info_global = None
class EnvironmentInfoNode(Node):
    # (Implementation from previous step)
    def __init__(self):
        super().__init__('environment_info_node')
        self.logger = self.get_logger()
        try:
            self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
            self.logger.info("EnvironmentInfoNode created and subscribed to /scan.")
        except Exception as e:
            self.logger.error("Failed to create ROS2 subscription: %s", e, exc_info=True)

    def lidar_callback(self, msg):
        global environment_info_global
        try:
            ranges = np.array(msg.ranges)
            finite_ranges = ranges[np.isfinite(ranges)]
            processed_info = np.array([-1.0, -1.0], dtype=np.float32)
            if len(finite_ranges) > 0:
                 processed_info = np.array([np.min(finite_ranges), np.mean(finite_ranges)], dtype=np.float32)
            environment_info_global = processed_info
        except Exception as e:
            self.logger.error("Error processing lidar scan: %s", e, exc_info=True)


# --- Pub/Sub Setup ---
# (Keep implementation from previous step)
# ...

# --- Cloud Storage Function (Definition only) ---
# (Keep implementation from previous step)
def save_graph_to_cloud_storage(graph, bucket_name, blob_name):
    # ... (implementation) ...
    pass

# --- Neptune API Client Placeholder ---
# (Keep implementation from previous step)
class NeptuneAPIClientPlaceholder:
    # ... (implementation) ...
    def __init__(self, api_endpoint="http://placeholder.api/graph", api_key="DUMMY_KEY"):
        self.logger = logging.getLogger(__name__)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.logger.info("Initialized NeptuneAPIClientPlaceholder for endpoint: %s", self.api_endpoint)
    # ... (rest of methods: get_object, query_objects, etc.) ...
    def get_object(self, object_id: str) -> dict | None: # Example
        self.logger.info("Simulating Neptune call: get_object(object_id='%s')", object_id)
        # ... return dummy data ...
        return {'id': object_id, 'state': [1,2,0,0], 'timestamp': datetime.now(timezone.utc).isoformat()} if random.random() < 0.9 else None
    # ... (implement other methods) ...
    def query_objects(self, criteria: dict) -> list[dict]: return [] # Placeholder
    def query_map_features(self, bounds: dict) -> list[dict]: return [] # Placeholder
    def upsert_object_state(self, object_id: str, state_vector: list | np.ndarray, timestamp_iso: str, **kwargs) -> bool:
         self.logger.info("Simulating Neptune upsert: %s", object_id)
         return random.random() < 0.98
    def update_object_semantics(self, object_id: str, properties: dict) -> bool:
         self.logger.info("Simulating Neptune semantic update: %s", object_id)
         return random.random() < 0.98


# --- Uncertainty Consultant ---
# (Keep UncertaintyConsultant class definition using BNN learner)
class UncertaintyConsultant:
    # ... (implementation from previous step) ...
    def __init__(self, bnn_learner_instance: BNNObjectBehaviorLearner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ):
        self.logger = logging.getLogger(__name__)
        if not isinstance(bnn_learner_instance, BNNObjectBehaviorLearner):
             self.logger.error("UncertaintyConsultant requires a BNNObjectBehaviorLearner instance.")
             raise TypeError("Invalid bnn_learner_instance type.")
        self.bnn_learner = bnn_learner_instance
        self.current_uncertainty_metrics = {}
        self.update_period_secs = 1.0 / update_frequency_hz if update_frequency_hz > 0 else float('inf')
        self.last_update_time = 0
        self.logger.info("UncertaintyConsultant initialized with update period %.2f s.", self.update_period_secs)

    def update_uncertainty(self, object_state_history: dict):
         current_time = time.time()
         if current_time - self.last_update_time < self.update_period_secs: return
         self.logger.debug("Running BNN uncertainty analysis...")
         overall_variance_sum = 0.0; num_predictions = 0; high_uncertainty_objects = []; object_variances = {}
         # Ensure history length matches learner requirement
         history_len_needed = self.bnn_learner.prediction_horizon or 10 # Get required len

         for obj_id, history_deque in object_state_history.items():
             if len(history_deque) >= history_len_needed: # Use >=
                 # Take only the most recent 'history_len_needed' states
                 history_list = list(history_deque)[-history_len_needed:]
                 history_array = np.array(history_list).reshape(1, history_len_needed, self.bnn_learner.state_dimension)
                 try:
                     mean_pred, var_pred = self.bnn_learner.predict(history_array)
                     if var_pred is not None:
                         pos_variance = var_pred[0, :, :2] # Positional variance (x,y)
                         avg_pos_var = tf.reduce_mean(pos_variance).numpy()
                         overall_variance_sum += avg_pos_var; num_predictions += 1; object_variances[obj_id] = avg_pos_var
                         if avg_pos_var > HIGH_UNCERTAINTY_THRESHOLD: high_uncertainty_objects.append(obj_id)
                 except Exception as e: self.logger.error("Error during BNN prediction for object %s: %s", obj_id, e, exc_info=True)

         avg_overall_variance = overall_variance_sum / num_predictions if num_predictions > 0 else 0.0
         self.current_uncertainty_metrics = {
             "average_positional_variance": avg_overall_variance,
             "high_uncertainty_ids": high_uncertainty_objects,
             "object_variances": object_variances }
         self.last_update_time = current_time
         self.logger.info("Uncertainty metrics updated: AvgVar=%.4f, HighUncertainty=%d obj", avg_overall_variance, len(high_uncertainty_objects))
         self.logger.debug("Detailed metrics: %s", self.current_uncertainty_metrics)

    def get_metrics(self) -> dict: return self.current_uncertainty_metrics


# --- Robot Navigator ---
# (Keep RobotNavigator definition, ensuring it uses Neptune client)
class RobotNavigator:
    # ... (implementation from previous step, uses self.neptune_client) ...
     def __init__(self, object_behavior_learner, neptune_client, max_retries=3, initial_delay=1):
          self.logger = logging.getLogger(__name__)
          self.object_learner = object_behavior_learner # Needed? Maybe just client? Keep for now.
          self.neptune_client = neptune_client
          self.robot_state = None
          self.environment_info = None
          self.gemini_model = None # Init Gemini model...
          self.max_retries = max_retries
          self.initial_delay = initial_delay
          self.planner = None # Planner will be set after initialization
          self.current_cautiousness = 0.0 # Example state for modulation
          # ... (Init Gemini Model) ...
          try:
              if gemini_configured: self.gemini_model = google.generativeai.GenerativeModel('gemini-pro')
          except Exception as e: self.logger.error("Failed to init Gemini: %s", e)

     def set_cautiousness(self, uncertainty_metric):
          """Example method to adjust behavior based on uncertainty."""
          # Simple scaling, needs tuning
          self.current_cautiousness = np.clip(uncertainty_metric * 2.0, 0.0, 1.0) # Scale variance to 0-1 range
          self.logger.debug("Navigator cautiousness set to %.2f", self.current_cautiousness)

     def get_cautiousness(self):
          return self.current_cautiousness
     # ... (Rest of methods: update_context, _build_gemini_prompt using Neptune, get_gemini_..., parse_..., map_actions_..., execute_..., run_navigation) ...
     # Ensure map_actions_to_robot_commands uses self.planner.plan_path(...)


# --- Path Planner (RRT*) ---
# (Keep PathPlannerRRTStar definition, with uncertainty adjustments)
Obstacle = namedtuple("Obstacle", ["x", "y", "radius"]) # Define Obstacle if used for static list
class Node: # Keep Node class definition
    def __init__(self, x, y, cost=0.0, parent=None): self.x=x; self.y=y; self.cost=cost; self.parent=parent
    def __eq__(self, other): return self.x == other.x and self.y == other.y
    def __hash__(self): return hash((self.x, self.y))
    def __repr__(self): return f"Node({self.x:.2f}, {self.y:.2f}, cost={self.cost:.2f})"

class PathPlannerRRTStar:
    # ... (Implementation from previous step, including __init__, adjust_safety_margins, _is_collision using margins) ...
     def __init__(self, behavior_learner, neptune_client, config_space_bounds, step_size=STEP_SIZE, max_iter=MAX_ITERATIONS, goal_sample_rate=GOAL_SAMPLE_RATE, search_radius=SEARCH_RADIUS, robot_radius=ROBOT_RADIUS):
          self.logger = logging.getLogger(__name__)
          self.behavior_learner = behavior_learner
          self.neptune_client = neptune_client
          self.min_x, self.max_x, self.min_y, self.max_y = config_space_bounds
          self.step_size = step_size; self.max_iter = max_iter; self.goal_sample_rate = goal_sample_rate; self.search_radius = search_radius
          self.robot_radius = robot_radius
          self.object_base_radius = OBJECT_BASE_RADIUS
          self.uncertainty_margins = {}
          self.default_uncertainty_margin = 0.0
          self.nodes = []; self.kdtree = None; self.start_node = None; self.goal_node = None
          self.logger.info(f"RRT* Planner Initialized. Using Neptune client. KDTree: {USE_KDTREE}")

     def adjust_safety_margins(self, average_variance: float, object_variances: dict):
          default_std_dev = math.sqrt(max(0, average_variance))
          self.default_uncertainty_margin = default_std_dev * PLANNER_SAFETY_FACTOR
          # self.logger.debug("Default uncertainty margin: %.3f", self.default_uncertainty_margin)
          self.uncertainty_margins.clear()
          for obj_id, var in object_variances.items():
               margin = math.sqrt(max(0, var)) * PLANNER_SAFETY_FACTOR
               self.uncertainty_margins[obj_id] = margin
               # self.logger.debug("Margin for %s: %.3f", obj_id, margin)

     def _is_collision(self, node1, node2, dynamic_obstacles_pred: dict, static_features: list):
          p1 = np.array([node1.x, node1.y]); p2 = np.array([node2.x, node2.y])
          segment_vec = p2 - p1; segment_len = np.linalg.norm(segment_vec)
          if segment_len < 1e-6: return False

          # Static Feature Check (Placeholder - needs geometry checks)
          for feature in static_features:
               # TODO: Implement geometry checks (point-line, point-polygon)
               pass # Placeholder - Assume no static collisions for now

          # Dynamic Obstacle Check (Using Uncertainty Margins)
          num_steps = max(2, int(segment_len / (self.step_size * 0.5)))
          for i in range(num_steps + 1):
               t_interp = i / num_steps
               robot_pos = p1 + t_interp * segment_vec
               for obj_id, trajectory in dynamic_obstacles_pred.items():
                    # Simplified: Check against all predicted points in trajectory
                    for (_time, obs_x, obs_y, _obs_base_radius) in trajectory:
                         margin = self.uncertainty_margins.get(obj_id, self.default_uncertainty_margin)
                         effective_obs_radius = self.object_base_radius + margin # Use base radius + margin
                         obs_center = np.array([obs_x, obs_y])
                         total_radius_sq = (self.robot_radius + effective_obs_radius)**2
                         dist_sq = np.sum((robot_pos - obs_center)**2)
                         if dist_sq <= total_radius_sq:
                              # self.logger.debug("Collision: Robot pos %s near DynObj %s (radius %.2f + margin %.2f)", robot_pos, obj_id, self.object_base_radius, margin)
                              return True
          return False

     # ... (Rest of PathPlannerRRTStar methods: get_goal_position, predict_dynamic_obstacles, plan_path, helpers) ...
     # Ensure predict_dynamic_obstacles uses self.behavior_learner (the realtime one)
     # Ensure plan_path calls get_goal_position (uses neptune), predict_dynamic_obstacles, queries neptune for static_features, and uses _is_collision correctly.


# --- Federated Learning Client Placeholder ---
# (Keep FederatedLearningClient class definition - unchanged from previous step)
class FederatedLearningClient:
     # ... (implementation from previous step) ...
     def __init__(self, keras_model, fl_server_url=FL_SERVER_URL): # Use config
          self.logger = logging.getLogger(__name__)
          self.local_model = keras_model
          self.server_url = fl_server_url
          self.tff_available = TFF_AVAILABLE
          self.initial_weights = None
          if TFF_AVAILABLE: # Get weights only if TFF might be used
               try: self.initial_weights = [w.numpy() for w in self.local_model.weights]
               except Exception as e: self.logger.error("Could not get initial weights for FL: %s", e)
          self.logger.info("FL Client initialized. TFF Available: %s", self.tff_available)
     # ... (rest of methods) ...


# --- Placeholder Functions ---
# (Keep placeholder function definitions, add logging if desired)
def get_user_command(): logger.debug("Checking for user command..."); return None
def get_robot_state(): return np.array([random.uniform(1,19), random.uniform(1,14), 0.0, 0.0]) # Random start state
def get_environment_info_for_combiner(): global environment_info_global; return environment_info_global.astype(np.float32) if environment_info_global is not None else np.zeros(2, dtype=np.float32)
def process_lidar_data_for_obstacles(lidar_data): return np.array([np.mean(lidar_data if lidar_data is not None else [-1.0])], dtype=np.float32)
def smooth_path(waypoints): logger.debug("Smoothing path (Placeholder)..."); return waypoints
def execute_path(path, cautiousness=0.0): # Added cautiousness example
    logger.info("Executing path (%d waypoints)... Cautiousness=%.2f", len(path) if path else 0, cautiousness)
    # TODO: Use cautiousness to modulate speed/control params
    if path is None: return
    for i, waypoint in enumerate(path): logger.debug(" -> Waypoint %d: %s", i+1, waypoint); time.sleep(0.1 * (1 + cautiousness)) # Slower if cautious

def detect_objects(frame, detector): # Keep simplified version for now
     # ... (implementation from previous step) ...
     detections = [{'id': f'obj_{int(time.time()*10 % 1000) + i}', 'measurement': np.array([random.uniform(1,19), random.uniform(1,14)]), 'confidence': 0.85, 'class': 'sim_obj'} for i in range(random.randint(1,4))]
     return detections

# Gemini Object Query Placeholders (Add logging)
def object_is_unexplored(object_id, graph_cache): return not graph_cache.nodes.get(object_id, {}).get('explored', False) # Use cache? Or query Neptune? Needs decision.
def object_confidence_is_low(object_id, graph_cache): return graph_cache.nodes.get(object_id, {}).get('gemini_confidence', 0.0) < 0.7
def capture_object_image(frame, detection): return frame[0:50, 0:50] # Placeholder
def query_gemini_for_object(image, question): logger.info("Simulating Gemini Vision query..."); return {'label': 'sim_cup', 'description': 'Simulated description.', 'confidence': 0.9}
def update_object_graph_with_gemini(neptune_client, object_id, gemini_response): logger.info("Simulating Neptune update with Gemini info for %s", object_id); neptune_client.update_object_semantics(object_id, gemini_response) # Use client
def should_explore_object(object_id, graph_cache): return False # Placeholder
def explore_object(object_id, navigator): logger.info("Simulating exploration command for %s", object_id); # navigator.run_navigation(...) # Example


# ==============================================================================
# === Main Robot Loop ==========================================================
# ==============================================================================
def robot_loop(neptune_client_instance): # Pass client instance
    loop_logger = logging.getLogger("RobotLoop")
    loop_logger.info("--- Initializing Robot Loop Components ---")

    # Initialization Block
    try:
        behavior_learner = ObjectBehaviorLearner(state_dimension=KALMAN_STATE_DIM, prediction_horizon=history_length) # Use global constant
        bnn_learner = BNNObjectBehaviorLearner(state_dimension=KALMAN_STATE_DIM, num_mc_samples=BNN_MC_SAMPLES, prediction_horizon=history_length)
        uncertainty_consultant = UncertaintyConsultant(bnn_learner_instance=bnn_learner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ)

        # Combiner setup (ensure dims match reality)
        env_dim = 2; robot_dim = KALMAN_STATE_DIM; obstacle_dim = 1
        context_dim = env_dim + robot_dim + obstacle_dim
        transformer_output_dim = KALMAN_STATE_DIM * behavior_learner.prediction_horizon # If predicting horizon
        # Adjust based on actual transformer output shape after flattening
        transformer_output_dim_flat = KALMAN_STATE_DIM # If using only last prediction state? Needs clarification from model definition. Let's assume last state for now.

        combiner_output_dim = 2 # e.g., vx, vy command or target x,y
        combiner_prediction_horizon = 10
        combiner = PredictionCombiner(transformer_output_dim_flat, context_dim, combiner_output_dim, combiner_prediction_horizon)
        combiner.compile(optimizer='adam', loss='mse')

        navigator = RobotNavigator(behavior_learner, neptune_client_instance) # Pass client
        planner = PathPlannerRRTStar(behavior_learner, neptune_client_instance, CONFIG_SPACE_BOUNDS) # Pass client
        navigator.planner = planner # Give navigator access to planner

        fl_client = FederatedLearningClient(behavior_learner.model) # Train the non-BNN model?

        trackers = {}
        object_state_history = {}
        history_length = 10

        loop_logger.info("Initializing ROS2 node...")
        # rclpy.init(args=None) # Init should happen outside if loop is called multiple times
        environment_info_node = EnvironmentInfoNode()

        loop_logger.info("Opening video capture...")
        capture = cv2.VideoCapture(0)
        if not capture.isOpened(): raise IOError("Cannot open webcam")

        robot_id = f'robot_{random.randint(1000, 9999)}' # Example dynamic ID
        frame_count = 0
        last_fl_check_time = time.time()
        fl_check_interval = 60

        loop_logger.info("Initialization complete. Starting main loop for robot: %s", robot_id)

    except Exception as e:
        loop_logger.critical("Initialization failed: %s. Exiting.", e, exc_info=True)
        # Cleanup attempt
        if 'capture' in locals() and isinstance(capture, cv2.VideoCapture) and capture.isOpened(): capture.release()
        # if rclpy.ok(): rclpy.shutdown() # Shutdown only if init was successful
        return # Exit function if init fails


    # --- Main Loop ---
    try:
        while rclpy.ok():
            loop_start_time = time.time()
            rclpy.spin_once(environment_info_node, timeout_sec=0.01)

            ret, frame = capture.read()
            if not ret:
                loop_logger.warning("Failed to read frame. Skipping.")
                time.sleep(0.1)
                continue
            frame_count += 1

            try: # Wrap main iteration logic
                detections = detect_objects(frame, object_detector)
                current_robot_state = get_robot_state() # Includes pose [x,y,...]
                current_environment_info_comb = get_environment_info_for_combiner()
                current_obstacles_comb = process_lidar_data_for_obstacles(environment_info_global)

                processed_object_ids = set()
                realtime_object_predictions = {}
                object_actual_states = {}
                current_frame_fl_data = []

                # --- Tracking, Prediction, State Update ---
                for detection in detections:
                     try:
                        object_id = detection.get('id')
                        measurement = detection.get('measurement')
                        if object_id is None or measurement is None: continue
                        processed_object_ids.add(object_id)

                        if object_id not in trackers:
                            initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
                            trackers[object_id] = KalmanTracker(initial_state)
                            object_state_history[object_id] = deque(maxlen=history_length)
                        kalman_tracker = trackers[object_id]
                        predicted_state_kf = kalman_tracker.predict()
                        updated_state_kf = kalman_tracker.update(measurement)
                        if updated_state_kf is None: continue

                        current_state_flat = updated_state_kf.flatten()
                        object_actual_states[object_id] = current_state_flat
                        current_timestamp_iso = datetime.now(timezone.utc).isoformat()

                        # Update local cache & Neptune
                        behavior_learner.update_local_cache(object_id, updated_state_kf, current_timestamp_iso)
                        neptune_client_instance.upsert_object_state(object_id, current_state_flat, current_timestamp_iso, label=detection.get('class')) # Send update

                        # History
                        object_state_history[object_id].append(current_state_flat)
                        if len(object_state_history[object_id]) == history_length:
                             current_frame_fl_data.append(list(object_state_history[object_id]))

                        # Real-time Prediction (Point Estimate)
                        if len(object_state_history[object_id]) >= history_length:
                             history_array = np.array(list(object_state_history[object_id])).reshape(1, history_length, KALMAN_STATE_DIM)
                             # Use the standard learner's predict
                             pred_tensor = behavior_learner.predict(history_array)
                             if pred_tensor is not None:
                                 # Assuming prediction is for the last step input
                                 realtime_object_predictions[object_id] = pred_tensor[0, -1, :].numpy() # Get last prediction

                     except (ValueError, Exception) as e_det:
                          loop_logger.error("Error processing detection %s: %s", detection.get('id', 'N/A'), e_det, exc_info=True)

                # Remove Old Trackers
                # ... (logic) ...
                lost_track_ids = set(trackers.keys()) - processed_object_ids
                for lost_id in lost_track_ids:
                    loop_logger.info("Object %s lost track.", lost_id)
                    if lost_id in trackers: del trackers[lost_id]
                    if lost_id in object_state_history: del object_state_history[lost_id]


                # --- Uncertainty Analysis ---
                try:
                    uncertainty_consultant.update_uncertainty(object_state_history)
                except Exception as e_uncert: loop_logger.error("Error updating uncertainty: %s", e_uncert, exc_info=True)

                # --- Get & Use Uncertainty ---
                metrics = uncertainty_consultant.get_metrics()
                avg_variance = metrics.get("average_positional_variance", 0.0)
                object_variances = metrics.get("object_variances", {})
                try:
                    planner.adjust_safety_margins(avg_variance, object_variances)
                    navigator.set_cautiousness(avg_variance) # Example
                except Exception as e_adjust: loop_logger.error("Error adjusting margins/cautiousness: %s", e_adjust, exc_info=True)

                # --- Combiner Prediction ---
                waypoints = None
                # Use realtime_object_predictions (point estimates) for combiner input
                if realtime_object_predictions:
                    try:
                        # --- Prepare combiner inputs (CRITICAL: ensure shapes/logic are correct) ---
                        # Example: Use average of point estimates
                        if object_actual_states and realtime_object_predictions:
                             pred_values = list(realtime_object_predictions.values())
                             avg_realtime_pred = np.mean(np.array(pred_values), axis=0) if pred_values else np.zeros(KALMAN_STATE_DIM)
                             # Confidence might be based on tracker confidence or prediction stability (needs impl)
                             dummy_confidence = np.array([0.8], dtype=np.float32) # Placeholder confidence

                             combiner_input_list = [
                                 tf.constant(avg_realtime_pred.reshape(1, -1), dtype=tf.float32), # Check shape needed by combiner
                                 tf.constant(dummy_confidence.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_environment_info_comb.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_robot_state.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_obstacles_comb.reshape(1, -1), dtype=tf.float32) ]
                             waypoints_tensor = combiner(combiner_input_list, training=False)
                             if waypoints_tensor is not None: waypoints = waypoints_tensor.numpy()[0]
                    except Exception as e_comb: loop_logger.error("Error during prediction combination: %s", e_comb, exc_info=True)

                # --- Navigation Decision ---
                user_command = get_user_command()
                if user_command:
                    loop_logger.info("Processing user command: '%s'", user_command)
                    try:
                        navigator.update_context(current_robot_state, current_environment_info_comb)
                        navigator.run_navigation(user_command) # Uses planner with adjusted margins
                    except Exception as e_nav: loop_logger.error("Error during command execution flow: %s", e_nav, exc_info=True)
                elif waypoints is not None:
                    try:
                        smoothed_path = smooth_path(waypoints)
                        # Pass cautiousness derived from uncertainty
                        execute_path(smoothed_path, cautiousness=navigator.get_cautiousness())
                    except Exception as e_path: loop_logger.error("Error executing combiner path: %s", e_path, exc_info=True)
                else:
                    loop_logger.debug("Idle.")
                    time.sleep(0.1)

                # --- Federated Learning ---
                current_time = time.time()
                if TFF_AVAILABLE and current_time - last_fl_check_time > fl_check_interval:
                    loop_logger.info("Running periodic Federated Learning check...")
                    if current_frame_fl_data:
                         try:
                             fl_client.run_fl_round(current_frame_fl_data)
                         except Exception as e_fl: loop_logger.error("Error during FL round: %s", e_fl, exc_info=True)
                    else: loop_logger.info("Skipping FL round: No new data.")
                    last_fl_check_time = current_time

            except Exception as e_iter:
                 loop_logger.error("Unhandled error in main loop iteration: %s", e_iter, exc_info=True)
                 time.sleep(1)

            # --- Loop Timing ---
            loop_duration = time.time() - loop_start_time
            # loop_logger.debug("Loop iteration %d finished in %.3f seconds", frame_count, loop_duration)


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
                 try: environment_info_node.destroy_node()
                 except Exception as e_ros_destroy: loop_logger.error("Error destroying ROS node: %s", e_ros_destroy)
            loop_logger.info("Shutting down ROS2...")
            try: rclpy.shutdown()
            except Exception as e_ros_shutdown: loop_logger.error("Error shutting down ROS2: %s", e_ros_shutdown)
        loop_logger.info("Cleanup complete.")


# --- Main Execution Guard ---
if __name__ == '__main__':
    logger.info("=================================================================")
    logger.info("=== Starting Robot Application (Neptune + FL + BNN Consultant) ===")
    logger.info("=================================================================")
    neptune_client_main = None
    try:
        # Instantiate the Neptune client placeholder (dependency)
        neptune_client_main = NeptuneAPIClientPlaceholder(api_endpoint=NEPTUNE_API_ENDPOINT)
        # Run the main loop, passing the client
        robot_loop(neptune_client_instance=neptune_client_main)
    except SystemExit as e:
         logger.warning("Application exited with code %s", e.code)
    except Exception as e:
        logger.critical("Application crashed at top level: %s", e, exc_info=True)
    finally:
        logger.info("========================================")
        logger.info("=== Robot Application Finished        ===")
        logger.info("========================================")
        logging.shutdown() # Ensure all logs are flushed
