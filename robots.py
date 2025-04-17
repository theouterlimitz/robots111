# --- Imports ---
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
from datetime import datetime, timezone, timedelta # Added timezone
from google.cloud import storage
import logging
import time
import sys
import math
import random
import re # Added for Gemini response parsing
from collections import namedtuple, deque

# Optional KDTree
try:
    from scipy.spatial import KDTree
    USE_KDTREE = True
except ImportError:
    USE_KDTREE = False
    print("Warning: scipy.spatial.KDTree not found. Falling back to slower nearest neighbor search.")

# Google Cloud Exceptions
from google.api_core import exceptions as google_api_exceptions
from google.cloud import exceptions as google_cloud_exceptions

# --- Logging Configuration ---
# (Keep logging setup from previous step - configure root logger, handlers)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
log_level = logging.INFO
root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    # File Handler (optional but recommended)
    log_file = 'robot_app.log'
    # Use RotatingFileHandler for production
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # General logger

# --- Configuration ---
# (Keep configuration variables and basic validation)
logger.info("Loading configuration...")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
YOLO_CFG_PATH = "yolov3.cfg"
YOLO_WEIGHTS_PATH = "yolov3.weights"
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", 'your-gcp-project-id')
PUBSUB_TOPIC_ID = 'object_exploration'
PUBSUB_SUBSCRIPTION_ID = 'robot_subscription'
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'your-bucket-name')
GCS_BLOB_NAME = 'graph_data.json' # Maybe archive old graphs here now?
CONFIG_SPACE_BOUNDS = (0, 20, 0, 15)
STEP_SIZE = 0.5
MAX_ITERATIONS = 5000
GOAL_SAMPLE_RATE = 0.1
SEARCH_RADIUS = 1.5
ROBOT_RADIUS = 0.3
# ... (validate configs) ...

# --- Configure Gemini API ---
# (Keep Gemini config logic)
try:
    if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
        google.generativeai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API configured.")
    else: logger.error("Skipping Gemini API configuration: Missing/placeholder key.")
except Exception as e: logger.critical("Failed to configure Gemini API: %s", e, exc_info=True)

# --- Object Detection ---
# (Keep YOLO loading logic with error handling)
logger.info("Loading Object Detector (YOLO)...")
object_detector = None
try:
    # ... (loading logic) ...
    if not os.path.exists(YOLO_CFG_PATH): raise FileNotFoundError(f"YOLO Cfg not found: {YOLO_CFG_PATH}")
    if not os.path.exists(YOLO_WEIGHTS_PATH): raise FileNotFoundError(f"YOLO Weights not found: {YOLO_WEIGHTS_PATH}")
    object_detector = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    logger.info("Object Detector loaded successfully.")
except (cv2.error, FileNotFoundError) as e:
    logger.critical("Failed to load YOLO model: %s", e, exc_info=True)
    sys.exit(1)


# --- Kalman Filter ---
KALMAN_STATE_DIM = 4
KALMAN_MEASURE_DIM = 2
# (Keep KalmanTracker class definition - unchanged)
class KalmanTracker:
     # ... (implementation from previous step) ...
     def __init__(self, initial_state):
        self.logger = logging.getLogger(__name__)
        if initial_state is None or initial_state.shape[0] != KALMAN_STATE_DIM:
             msg = f"Initial state must have dimension {KALMAN_STATE_DIM}, got shape {initial_state.shape if initial_state is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)
        # ... rest of Kalman init ...

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
# (Keep TransformerBlock class definition - unchanged)
class TransformerBlock(tf.keras.layers.Layer):
    # ... (implementation from previous step) ...
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model # Store params
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        # ... rest of MHA, LayerNorm, FFN, Dropout init ...
        # Use key_dim=embed_dim // num_heads if embed_dim is divisible by num_heads
        # Or ensure d_model is divisible by num_heads
        # For simplicity, using key_dim=d_model here, might need adjustment
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads == 0 else self.d_model)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [Dense(self.dff, activation='relu'), Dense(self.d_model)]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, x, training, mask=None): # Added mask parameter
        # Ensure mask shape compatibility if using masks
        # If using self-attention, query, key, value are the same (x)
        attn_output, attn_weights = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# (Keep ObjectBehaviorLearner class definition - graph update logic changes later)
class ObjectBehaviorLearner:
    # ... (Keep __init__, train, predict methods) ...
    def __init__(self, prediction_horizon=10, state_dimension=KALMAN_STATE_DIM, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(__name__)
        self.prediction_horizon = prediction_horizon
        self.state_dimension = state_dimension
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        # Local graph now acts as a *cache* of recently seen/relevant objects
        # Persistence handled by Neptune client calls
        self.local_object_graph_cache = nx.Graph()
        self.logger.info("ObjectBehaviorLearner initialized (local graph acts as cache).")
        # ... (Keep TF model building and compile) ...
        try:
             if self.d_model % self.num_heads != 0:
                 self.logger.warning("Transformer d_model (%d) is not divisible by num_heads (%d).", self.d_model, self.num_heads)
             # ... (rest of model building) ...
             input_layer = Input(shape=(None, self.state_dimension), name="transformer_input")
             x = Dense(d_model, name="input_embedding")(input_layer)
             for i in range(num_layers):
                 # Pass training=None initially, set dynamically during fit/predict
                 x = TransformerBlock(d_model, num_heads, dff, name=f"transformer_block_{i}")(x, training=None)
             output_layer = TimeDistributed(Dense(self.state_dimension), name="output_dense")(x)
             self.model = Model(inputs=input_layer, outputs=output_layer)
             self.model.compile(optimizer='adam', loss='mse')
             self.logger.info("Transformer Model compiled successfully.")
        except Exception as e:
             self.logger.critical("Failed to build or compile Transformer model: %s", e, exc_info=True)
             raise

    def train(self, past_states, target_states):
        self.logger.info("Starting Transformer training...")
        try:
            # Pass training=True to the model layers during fit
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            self.logger.info("Transformer training finished. Final validation loss: %s", history.history['val_loss'][-1])
        except tf.errors.InvalidArgumentError as e:
            self.logger.error("TF InvalidArgumentError during training: %s", e, exc_info=True)
        except Exception as e:
            self.logger.error("Unexpected error during Transformer training: %s", e, exc_info=True)

    def predict(self, past_states):
        self.logger.debug("Predicting with Transformer for %d samples", past_states.shape[0])
        try:
            # Pass training=False to the model layers during predict
            predictions = self.model.predict(past_states)
            return predictions
        except tf.errors.InvalidArgumentError as e:
            self.logger.error("TF InvalidArgumentError during prediction: %s", e, exc_info=True)
            return None
        except Exception as e:
             self.logger.error("Unexpected error during Transformer prediction: %s", e, exc_info=True)
             return None

    def update_local_cache(self, object_id, state, timestamp_iso):
        """Updates the local NetworkX graph cache."""
        state_values = state.flatten().tolist()
        if object_id not in self.local_object_graph_cache.nodes:
            self.local_object_graph_cache.add_node(object_id, state=state_values, timestamp=timestamp_iso)
            self.logger.debug("Added node %s to local cache", object_id)
        else:
            self.local_object_graph_cache.nodes[object_id]['state'] = state_values
            self.local_object_graph_cache.nodes[object_id]['timestamp'] = timestamp_iso
            # self.logger.debug("Updated node %s in local cache", object_id) # Too verbose?

    # Removed Pub/Sub methods - better handled centrally or passed publisher client


# --- Prediction Combiner (TensorFlow) ---
# (Keep PredictionCombiner definition - unchanged)
class PredictionCombiner(tf.keras.Model):
     # ... (implementation from previous step) ...
     def __init__(self, transformer_output_dim, context_dim, output_dim=KALMAN_STATE_DIM, prediction_horizon=10, use_confidence_weighting=True, dropout_rate=0.2):
        super(PredictionCombiner, self).__init__()
        self.logger = logging.getLogger(__name__)
        # ... rest of init ...
        self.output_dim = output_dim
        self.prediction_horizon = prediction_horizon
        self.use_confidence_weighting = use_confidence_weighting
        self.dropout_rate = dropout_rate
        self.dense1 = Dense(128, activation='relu', name="combiner_dense1")
        self.dense2 = Dense(64, activation='relu', name="combiner_dense2")
        self.dense3 = Dense(self.output_dim * self.prediction_horizon, name="combiner_output_dense")
        self.layer_norm = LayerNormalization(epsilon=1e-6, name="combiner_layernorm")
        self.dropout = Dropout(self.dropout_rate, name="combiner_dropout")
        self.logger.info("Prediction Combiner Initialized.")

     def call(self, inputs, training=None):
        try:
            # ... (implementation: flatten, weight, concat, dense, reshape) ...
             transformer_predictions, transformer_confidences, environment_info, robot_state, obstacles = inputs
             # ... (flattening logic using tf.shape and Flatten()) ...
             transformer_predictions_flat = Flatten()(transformer_predictions) if len(tf.shape(transformer_predictions)) > 2 else transformer_predictions
             # ... (flatten others) ...
             transformer_confidences_flat = Flatten()(transformer_confidences) if len(tf.shape(transformer_confidences)) > 1 else transformer_confidences
             environment_info_flat = Flatten()(environment_info) if len(tf.shape(environment_info)) > 2 else environment_info
             robot_state_flat = Flatten()(robot_state) if len(tf.shape(robot_state)) > 2 else robot_state
             obstacles_flat = Flatten()(obstacles) if len(tf.shape(obstacles)) > 1 else obstacles

             weighted_preds = transformer_predictions_flat
             if self.use_confidence_weighting and transformer_confidences_flat is not None:
                 confidences_exp = tf.expand_dims(transformer_confidences_flat, axis=-1)
                 # Ensure broadcasting works or dimensions match after flattening
                 weighted_preds = transformer_predictions_flat * confidences_exp

             # Ensure all parts are tensors and flatten correctly before concat
             combined_input = Concatenate()([weighted_preds, environment_info_flat, robot_state_flat, obstacles_flat])

             # ... (norm, dropout, dense, reshape) ...
             combined_input = self.layer_norm(combined_input, training=training)
             combined_input = self.dropout(combined_input, training=training)
             combined_input = self.dense1(combined_input)
             combined_input = self.dense2(combined_input)
             waypoints_flat = self.dense3(combined_input)
             waypoints = Reshape((self.prediction_horizon, self.output_dim))(waypoints_flat)

             return waypoints
        except tf.errors.InvalidArgumentError as e:
             self.logger.error("TF InvalidArgumentError in Combiner call: %s", e, exc_info=True)
             return None
        except Exception as e:
             self.logger.error("Unexpected error in PredictionCombiner call: %s", e, exc_info=True)
             return None


# --- ROS2 Node ---
# (Keep EnvironmentInfoNode definition with internal logging/error handling)
environment_info_global = None
class EnvironmentInfoNode(Node):
    # ... (implementation from previous step with internal self.logger and try-except in callback) ...
    def __init__(self):
        super().__init__('environment_info_node')
        self.logger = self.get_logger()
        try:
            self.subscription = self.create_subscription(
                LaserScan, '/scan', self.lidar_callback, 10)
            self.logger.info("EnvironmentInfoNode created and subscribed to /scan.")
        except Exception as e:
            self.logger.error("Failed to create ROS2 subscription: %s", e, exc_info=True)
            # Consider raising or handling this failure

    def lidar_callback(self, msg):
        global environment_info_global
        try:
            # TODO: Implement robust Lidar processing
            ranges = np.array(msg.ranges)
            finite_ranges = ranges[np.isfinite(ranges)]
            processed_info = np.array([-1.0, -1.0], dtype=np.float32) # Default value
            if len(finite_ranges) > 0:
                 min_range = np.min(finite_ranges)
                 mean_range = np.mean(finite_ranges)
                 processed_info = np.array([min_range, mean_range], dtype=np.float32)

            environment_info_global = processed_info
            # self.logger.debug(f'Lidar processed: {environment_info_global}')
        except Exception as e:
            self.logger.error("Error processing lidar scan: %s", e, exc_info=True)
            # Optionally set global to None on error
            # environment_info_global = None


# --- Pub/Sub Setup ---
# (Keep Pub/Sub setup logic with refined error handling/logging)
# ... (implementation from previous step) ...
logger.info("Initializing Google Cloud Pub/Sub clients...")
publisher = None
subscriber = None
topic_path = None
subscription_path = None
try:
    # ... (init clients, paths) ...
    publisher = pubsub.PublisherClient()
    subscriber = pubsub.SubscriberClient()
    logger.info("Pub/Sub clients initialized.")
    topic_path = publisher.topic_path(GCP_PROJECT_ID, PUBSUB_TOPIC_ID)
    subscription_path = subscriber.subscription_path(GCP_PROJECT_ID, PUBSUB_SUBSCRIPTION_ID)
    # ... (create topic/sub with specific error catches) ...
    # Create topic
    try:
        publisher.create_topic(name=topic_path)
        logger.info("Topic %s created or already exists.", topic_path)
    except google_api_exceptions.AlreadyExists:
        logger.info("Topic %s already exists.", topic_path)
    except google_api_exceptions.PermissionDenied as e:
        logger.error("Permission denied creating topic %s: %s", topic_path, e)
    except Exception as e:
        logger.error("Failed to create topic %s: %s", topic_path, e, exc_info=True)

    # Create subscription
    try:
        subscriber.create_subscription(name=subscription_path, topic=topic_path)
        logger.info("Subscription %s created or already exists.", subscription_path)
    except google_api_exceptions.AlreadyExists:
        logger.info("Subscription %s already exists.", subscription_path)
    except google_api_exceptions.PermissionDenied as e:
        logger.error("Permission denied creating subscription %s: %s", subscription_path, e)
    except Exception as e:
        logger.error("Failed to create subscription %s: %s", subscription_path, e, exc_info=True)

except Exception as e:
    logger.error("Failed to initialize Google Cloud Pub/Sub clients: %s. Pub/Sub features may be unavailable.", e, exc_info=True)


# --- Cloud Storage Function (Definition only - runs in cloud) ---
# (Keep save_graph_to_cloud_storage function definition - unchanged)
def save_graph_to_cloud_storage(graph, bucket_name, blob_name):
     # ... (implementation from previous step with logging and specific GCS exceptions) ...
     func_logger = logging.getLogger(__name__)
     # ... (rest of function) ...

# --- Neptune API Client Placeholder ---
# (Keep NeptuneAPIClientPlaceholder class definition - unchanged)
class NeptuneAPIClientPlaceholder:
    # ... (implementation from previous step with logging and dummy data) ...
    def __init__(self, api_endpoint="http://placeholder.api/graph", api_key="DUMMY_KEY"):
        self.logger = logging.getLogger(__name__)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.logger.info("Initialized NeptuneAPIClientPlaceholder for endpoint: %s", self.api_endpoint)

    def get_object(self, object_id: str) -> dict | None:
        self.logger.info("Simulating Neptune call: get_object(object_id='%s')", object_id)
        if random.random() < 0.9:
            # ... (return dummy dict) ...
             dummy_state = [random.uniform(1, 19), random.uniform(1, 14), random.uniform(-1, 1), random.uniform(-1, 1)]
             dummy_timestamp = datetime.now(timezone.utc).isoformat()
             return {'id': object_id, 'state': dummy_state, 'timestamp': dummy_timestamp, 'label': 'dummy_label'}
        else:
            self.logger.warning("Simulating Neptune: Object '%s' not found.", object_id)
            return None

    def query_objects(self, criteria: dict) -> list[dict]:
        self.logger.info("Simulating Neptune call: query_objects(criteria=%s)", criteria)
        num_found = random.randint(0, 3)
        results = []
        for i in range(num_found):
             obj_id = f"sim_nearby_obj_{i}"
             dummy_state = [random.uniform(1, 19), random.uniform(1, 14), 0, 0]
             dummy_timestamp = datetime.now(timezone.utc).isoformat()
             results.append({'id': obj_id, 'state': dummy_state, 'timestamp': dummy_timestamp, 'label': 'nearby_dummy'})
        self.logger.info("Simulating Neptune query: Found %d objects.", len(results))
        return results

    def query_map_features(self, bounds: dict) -> list[dict]:
        self.logger.info("Simulating Neptune call: query_map_features(bounds=%s)", bounds)
        features = []
        # Simulate adding a static wall if bounds overlap
        if bounds.get('max_x', 0) > 5:
             features.append({'id': 'wall_static_1', 'type': 'wall', 'geometry': [(5,0), (5,15)]})
        self.logger.info("Simulating Neptune query: Found %d map features.", len(features))
        return features

    def upsert_object_state(self, object_id: str, state_vector: list | np.ndarray, timestamp_iso: str, location_geo: dict | None = None, label: str | None = None, other_props: dict | None = None) -> bool:
        self.logger.info("Simulating Neptune call: upsert_object_state(object_id='%s', state=%s, timestamp=%s)",
                         object_id, list(state_vector) if isinstance(state_vector, np.ndarray) else state_vector, timestamp_iso)
        success = random.random() < 0.98
        if not success: self.logger.error("Simulating Neptune upsert failure for object_id '%s'", object_id)
        return success

    def update_object_semantics(self, object_id: str, properties: dict) -> bool:
        self.logger.info("Simulating Neptune call: update_object_semantics(object_id='%s', properties=%s)",
                         object_id, properties)
        success = random.random() < 0.98
        if not success: self.logger.error("Simulating Neptune semantic update failure for object_id '%s'", object_id)
        return success


# --- Robot Navigator (Integrated with Neptune Client Placeholder) ---
class RobotNavigator:
    # Modified __init__
    def __init__(self, object_behavior_learner, neptune_client, max_retries=3, initial_delay=1): # Added neptune_client
        self.logger = logging.getLogger(__name__)
        self.object_learner = object_behavior_learner # Still needed for local predictions? Or just pass client? Decide. For now keep.
        self.neptune_client = neptune_client # Use the passed-in client
        self.robot_state = None
        self.environment_info = None
        self.gemini_model = None
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        # Keep Gemini Model Initialization
        try:
            if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_API_KEY_HERE":
                self.gemini_model = google.generativeai.GenerativeModel('gemini-pro')
                self.logger.info("Gemini Model (%s) initialized.", 'gemini-pro')
            else:
                 self.logger.error("Gemini API Key is missing or placeholder. Cannot initialize Gemini Model.")
        except Exception as e:
            self.logger.error("Error initializing Gemini Model: %s", e, exc_info=True)
            self.gemini_model = None

    # Keep update_context
    def update_context(self, robot_state, environment_info):
        self.robot_state = robot_state
        self.environment_info = environment_info

    # Keep format_environment_info (unchanged for now)
    def format_environment_info(self):
        # ... (implementation from previous step) ...
        if self.environment_info is None: return "No environment information available."
        try:
            return ", ".join(map(str, self.environment_info))
        except Exception as e:
            self.logger.error("Error formatting environment info: %s", e, exc_info=True)
            return "Error formatting environment data."


    # Modified _build_gemini_prompt to use Neptune client for graph context
    def _build_gemini_prompt(self, user_command):
        # Fetch relevant context from Neptune instead of formatting local graph
        graph_context_str = "No graph context available or error fetching."
        map_features_str = "No map features available or error fetching."
        try:
            # Define criteria (e.g., based on robot state location)
            # TODO: Get robot location from self.robot_state
            robot_loc = {'lat': 35.8, 'lon': -86.3} # Placeholder location
            radius_m = 50 # Example radius
            query_criteria = {'location_radius': {'lat': robot_loc['lat'], 'lon': robot_loc['lon'], 'radius_m': radius_m}}

            nearby_objects = self.neptune_client.query_objects(criteria=query_criteria)

            # Format the fetched objects concisely for the prompt
            if nearby_objects:
                object_summaries = [f"- ID: {obj.get('id')}, Label: {obj.get('label', 'N/A')}, State: {obj.get('state', [])[:2]}" # Show only x,y?
                                    for obj in nearby_objects]
                graph_context_str = "Nearby objects:\n" + "\n".join(object_summaries)
            else:
                graph_context_str = "No relevant objects found nearby."

            # Query map features (optional)
            # TODO: Define bounds based on robot location/task
            map_bounds = {'min_x': 0, 'max_x': 20, 'min_y': 0, 'max_y': 15} # Placeholder bounds
            map_features = self.neptune_client.query_map_features(bounds=map_bounds)
            if map_features:
                 map_features_str = "Relevant map features:\n" + json.dumps(map_features, indent=2)
            else:
                 map_features_str = "No relevant map features found."

        except Exception as e:
             self.logger.error("Error fetching context from Neptune Client for Gemini prompt: %s", e, exc_info=True)

        # Format other context
        env_info_str = self.format_environment_info()
        robot_state_str = str(self.robot_state)

        # Build the prompt using the fetched context
        prompt = f"""
        You are a robot navigation assistant... [Rest of prompt details]...

        Context:
        Robot State: {robot_state_str}
        Environment Info (e.g., sensor data): {env_info_str}
        Nearby Objects & Map Features (from Graph DB):
        {graph_context_str}
        {map_features_str}

        User Command: "{user_command}"

        Instructions:
        Provide a sequence of high-level actions... [Rest of prompt details]...
        """
        return prompt

    # Keep get_gemini_navigation_instructions (uses updated _build_gemini_prompt)
    def get_gemini_navigation_instructions(self, user_command):
         # ... (implementation from previous step with retries and logging) ...
         # Calls the now-modified _build_gemini_prompt
         if not self.gemini_model:
             self.logger.error("Cannot get Gemini instructions: Model not available.")
             return None
         prompt = self._build_gemini_prompt(user_command)
         # ... (rest of API call with retries) ...
         # ... Copy retry logic from previous version ...
         retries = 0
         delay = self.initial_delay
         while retries < self.max_retries:
             try:
                 # ... (call generate_content, check safety feedback) ...
                 response = self.gemini_model.generate_content(prompt)
                 if response.prompt_feedback.block_reason:
                      self.logger.error("Gemini request blocked due to safety settings: %s", response.prompt_feedback.block_reason)
                      return None
                 navigation_instructions_text = response.text
                 self.logger.info("Received Gemini response.")
                 return navigation_instructions_text
             # ... (Catch specific retryable/non-retryable exceptions) ...
             except (google_api_exceptions.ResourceExhausted,
                     google_api_exceptions.InternalServerError,
                     google_api_exceptions.ServiceUnavailable,
                     google_api_exceptions.DeadlineExceeded) as e:
                  retriable_error = True
                  error_type = type(e).__name__
                  self.logger.warning("Gemini API Error (%s): %s. Retrying (%d/%d) in %d s...", error_type, e, retries + 1, self.max_retries, delay)
             except (google_api_exceptions.PermissionDenied,
                     google_api_exceptions.InvalidArgument,
                     google_api_exceptions.GoogleAPIError) as e:
                  non_retriable_error = True
                  error_type = type(e).__name__
                  self.logger.error("Non-retryable Gemini API Error (%s): %s", error_type, e, exc_info=True)
                  return None # Exit loop
             except Exception as e: # Catch other potential errors
                 self.logger.error("Unexpected error calling Gemini API: %s", e, exc_info=True)
                 return None # Exit loop

             # Increment retry logic if it was a retryable error
             time.sleep(delay)
             retries += 1
             delay *= 2

         self.logger.error("Gemini API call failed after %d retries.", self.max_retries)
         return None


    # Keep parse_gemini_instructions (unchanged internally)
    def parse_gemini_instructions(self, instructions_text):
        # ... (implementation from previous step with logging) ...
        if not instructions_text: return None
        # ... (rest of implementation) ...
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', instructions_text, re.MULTILINE)
            # ... (rest of parsing logic) ...
            if match: json_str = match.group(1).strip()
            else:
                self.logger.warning("JSON markdown block not found. Attempting to parse entire response.")
                json_str = instructions_text.strip()
            instructions_list = json.loads(json_str)
            # ... (rest of validation logic) ...
            return valid_instructions # assuming valid_instructions list was populated
        except (json.JSONDecodeError, Exception) as e:
             self.logger.error("Error parsing Gemini instructions: %s", e, exc_info=True)
             return None


    # Keep map_actions_to_robot_commands (calls updated planner)
    def map_actions_to_robot_commands(self, instructions):
        # Needs access to the planner instance
        # This suggests planner might need to be passed in or be an attribute
        # For now, assume planner is accessible (e.g., self.planner if added to init)
        if not hasattr(self, 'planner'):
             self.logger.error("Path planner not available to RobotNavigator.")
             # Or instantiate it here if needed, passing the neptune client
             # self.planner = PathPlannerRRTStar(self.object_learner, self.neptune_client, CONFIG_SPACE_BOUNDS)

        robot_commands = []
        if instructions is None: return None
        self.logger.info("Mapping %d high-level instructions...", len(instructions))
        for i, instruction in enumerate(instructions):
            action = instruction.get("action")
            target = instruction.get("target")
            self.logger.debug("Mapping instruction %d: action=%s, target=%s", i+1, action, target)

            if action == "go_to":
                if target is None:
                    self.logger.warning("Skipping 'go_to' action: missing target.")
                    continue
                # --- Integrate Path Planner Call ---
                self.logger.info("Requesting path planning to target: %s", target)
                # Assuming self.planner exists and robot_state holds current pose
                # Also needs environment info (e.g., current lidar scan processed)
                # This env info might need to be passed into this function
                current_env_snapshot = get_environment_info_for_combiner() # Example
                waypoints = self.planner.plan_path(self.robot_state, target, current_env_snapshot) # Pass env snapshot
                if waypoints:
                    self.logger.info("Path planning successful, adding move commands.")
                    for wp in waypoints: robot_commands.append(("move_to", wp))
                else:
                    self.logger.error("Path planning failed for target: %s", target)
                    # Decide how to proceed - skip? report error?
            # ... (rest of action mapping from previous step) ...
            elif action == "scan_area": robot_commands.append(("scan", target))
            elif action == "approach": robot_commands.append(("approach", target))
            # ... etc ...
            else: self.logger.warning("Unrecognized action '%s' encountered during mapping.", action)

        self.logger.info("Generated %d low-level robot commands.", len(robot_commands))
        return robot_commands


    # Keep execute_commands (unchanged internally)
    def execute_commands(self, commands):
         # ... (implementation from previous step with logging) ...
         pass # Keep previous implementation

    # Keep run_navigation (unchanged internally)
    def run_navigation(self, user_command):
        # ... (implementation from previous step) ...
        pass # Keep previous implementation


# --- Path Planner (Integrated with Neptune Client Placeholder) ---
class PathPlannerRRTStar:
    # Modified __init__
    def __init__(self, behavior_learner, neptune_client, config_space_bounds, # Added neptune_client
                 step_size=STEP_SIZE, max_iter=MAX_ITERATIONS,
                 goal_sample_rate=GOAL_SAMPLE_RATE, search_radius=SEARCH_RADIUS,
                 robot_radius=ROBOT_RADIUS):
        self.logger = logging.getLogger(__name__)
        self.behavior_learner = behavior_learner # Still needed for Transformer predictions
        self.neptune_client = neptune_client # Use the client
        self.min_x, self.max_x, self.min_y, self.max_y = config_space_bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.robot_radius = robot_radius

        # Obstacles will now be fetched or updated based on sensor/Neptune queries
        self.static_map_features = [] # Store features queried from Neptune
        self.nodes = []
        self.kdtree = None
        self.start_node = None
        self.goal_node = None # Represents target position
        self.logger.info(f"RRT* Planner Initialized. Using Neptune client. KDTree: {USE_KDTREE}")

    # Removed update_static_obstacles - static features fetched dynamically

    # Modified get_goal_position to use Neptune client
    def get_goal_position(self, target) -> tuple | None:
        """Resolves target (ID or coords) to coordinates using Neptune client."""
        if isinstance(target, (list, tuple, np.ndarray)) and len(target) >= 2:
            return tuple(target[:2])
        elif isinstance(target, str):
             # Query Neptune for the object's state/location
             object_data = self.neptune_client.get_object(target)
             if object_data and 'state' in object_data and len(object_data['state']) >= 2:
                  coords = tuple(object_data['state'][:2])
                  self.logger.info("Target '%s' resolved to coords via Neptune: %s", target, coords)
                  return coords
             # Optional: Query for named locations if schema supports it
             # location_data = self.neptune_client.query_objects({'label': target, 'type': 'Location'}) ...
        self.logger.warning("Could not resolve target '%s' to coordinates via Neptune.", target)
        return None

    # Keep _distance, _get_random_point, _get_nearest_node, _find_near_nodes, _steer, _calculate_cost, _reconstruct_path (unchanged)
    def _distance(self, node1, node2):
        # ... (implementation) ...
        p1 = node1 if isinstance(node1, Node) else Node(node1[0], node1[1])
        p2 = node2 if isinstance(node2, Node) else Node(node2[0], node2[1])
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def _get_random_point(self):
        # ... (implementation) ...
        if random.random() < self.goal_sample_rate and self.goal_node:
            return (self.goal_node.x, self.goal_node.y)
        else:
            x = random.uniform(self.min_x, self.max_x)
            y = random.uniform(self.min_y, self.max_y)
            return (x, y)

    def _get_nearest_node(self, point):
        # ... (implementation using self.nodes and optional KDTree) ...
        if not self.nodes: return None
        if USE_KDTREE and self.kdtree:
            # ... (KDTree logic) ...
             distances, indices = self.kdtree.query(np.array(point), k=1)
             nearest_idx = indices[0] if isinstance(indices, np.ndarray) else indices
             return self.nodes[nearest_idx]

        else:
             # ... (Manual search logic) ...
             min_dist = float('inf')
             nearest = None
             for node in self.nodes:
                 dist = self._distance(node, point)
                 if dist < min_dist:
                     min_dist = dist
                     nearest = node
             return nearest


    def _find_near_nodes(self, target_node):
        # ... (implementation using self.nodes and optional KDTree) ...
        if not self.nodes: return []
        radius_sq = self.search_radius ** 2
        if USE_KDTREE and self.kdtree:
            # ... (KDTree logic) ...
             indices = self.kdtree.query_ball_point([target_node.x, target_node.y], r=self.search_radius)
             return [self.nodes[i] for i in indices]
        else:
             # ... (Manual search logic) ...
             near_nodes = []
             for node in self.nodes:
                  dist_sq = (node.x - target_node.x)**2 + (node.y - target_node.y)**2
                  if dist_sq <= radius_sq: near_nodes.append(node)
             return near_nodes


    def _steer(self, from_node, to_point):
        # ... (implementation) ...
         dist = self._distance(from_node, to_point)
         if dist <= self.step_size: return Node(to_point[0], to_point[1])
         else:
             theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
             new_x = from_node.x + self.step_size * math.cos(theta)
             new_y = from_node.y + self.step_size * math.sin(theta)
             return Node(new_x, new_y)

    def _calculate_cost(self, from_node, to_node):
         return from_node.cost + self._distance(from_node, to_node)

    def _reconstruct_path(self, goal_node_final):
         # ... (implementation) ...
         path = []
         current = goal_node_final
         while current is not None:
             path.append((current.x, current.y))
             current = current.parent
         return path[::-1]


    # Keep predict_dynamic_obstacles (still uses local behavior_learner)
    def predict_dynamic_obstacles(self, robot_pose, prediction_horizon_secs, time_step):
        self.logger.debug("Predicting dynamic obstacles using local Transformer...")
        # TODO: Implement prediction logic using self.behavior_learner
        # Needs access to object state history
        # For now, return empty placeholder
        return {}

    # Modified _is_collision to use fetched map features and dynamic predictions
    def _is_collision(self, node1, node2, dynamic_obstacles_pred, static_features):
        """
        Checks collision against static map features and predicted dynamic obstacles.
        """
        p1 = np.array([node1.x, node1.y])
        p2 = np.array([node2.x, node2.y])
        # ... (rest of collision check logic from previous RRT* implementation) ...
        # Modify static check to iterate through `static_features` queried from Neptune
        # Modify dynamic check based on `dynamic_obstacles_pred` structure

        # --- Static Feature Check ---
        # TODO: Implement collision checking against different geometry types from Neptune
        for feature in static_features:
             geom_type = feature.get('type')
             geometry = feature.get('geometry')
             if geom_type == 'wall' and isinstance(geometry, list) and len(geometry) == 2:
                  # Check collision with line segment wall
                  wall_p1 = np.array(geometry[0])
                  wall_p2 = np.array(geometry[1])
                  # ... (Line segment intersection check or distance check) ...
                  # Simplified: Treat wall as thick line? Check distance? Needs geometry library.
                  pass # Placeholder for wall collision
             elif geom_type == 'region' and isinstance(geometry, list):
                  # Check if path segment intersects polygon region (if region is obstacle)
                  pass # Placeholder for polygon collision
             elif geom_type == 'obstacle_circle': # Example
                  obs_center = np.array([geometry['x'], geometry['y']])
                  obs_radius = geometry['radius']
                  # ... (Use circle collision logic from previous implementation) ...

        # --- Dynamic Obstacle Check ---
        # ... (Keep dynamic check logic from previous RRT* implementation) ...
        # ... (Needs refinement regarding timing) ...

        return False # Placeholder: Assume no collision for now


    # Modified plan_path method
    def plan_path(self, start_pose, goal_target, environment_info):
        """
        Plans a path using RRT*, fetching map features from Neptune.

        Args:
            start_pose (tuple): (x, y) or (x, y, theta) of the robot.
            goal_target (str or tuple): Object ID, named location, or (x, y) coordinates.
            environment_info: Current processed sensor data (e.g., Lidar for local map/obstacles).
                               This might be used to *supplement* Neptune map features.

        Returns:
            list or None: A list of (x, y) waypoints, or None if no path found.
        """
        start_time = time.time()
        self.logger.info("Starting RRT* planning from %s to target '%s'", start_pose[:2], goal_target)

        # --- Initialization ---
        self.start_node = Node(start_pose[0], start_pose[1], cost=0.0, parent=None)
        goal_pos = self.get_goal_position(goal_target) # Uses Neptune client
        if goal_pos is None:
            self.logger.error("RRT* Error: Goal position could not be determined.")
            return None
        self.goal_node = Node(goal_pos[0], goal_pos[1])

        self.nodes = [self.start_node]
        if USE_KDTREE: self.kdtree = KDTree([[self.start_node.x, self.start_node.y]])
        best_goal_node = None

        # --- Fetch Static Map Features from Neptune ---
        # Define bounds around start/goal or based on expected path area
        # TODO: Define bounds dynamically
        query_bounds = {'min_x': min(start_pose[0], goal_pos[0]) - 5, 'max_x': max(start_pose[0], goal_pos[0]) + 5,
                        'min_y': min(start_pose[1], goal_pos[1]) - 5, 'max_y': max(start_pose[1], goal_pos[1]) + 5}
        static_features = self.neptune_client.query_map_features(bounds=query_bounds)
        self.logger.info("Fetched %d static map features from Neptune.", len(static_features))
        # TODO: Process environment_info (e.g., Lidar) to add *local*, real-time obstacles
        # to the static_features list or handle them separately in collision check.

        # --- Predict Dynamic Obstacles ---
        dynamic_preds = self.predict_dynamic_obstacles(start_pose, 5.0, 0.5) # Uses local TF model

        # --- RRT* Main Loop ---
        for i in range(self.max_iter):
            # ... (Sampling, Nearest, Steer logic remains the same) ...
            rnd_point = self._get_random_point()
            nearest_node = self._get_nearest_node(rnd_point)
            if nearest_node is None: continue
            new_potential_node = self._steer(nearest_node, rnd_point)


            # Pass BOTH static features and dynamic predictions to collision check
            if not self._is_collision(nearest_node, new_potential_node, dynamic_preds, static_features):
                 # ... (Find near nodes, choose parent logic remains the same) ...
                 near_nodes = self._find_near_nodes(new_potential_node)
                 min_cost_node = nearest_node
                 min_new_cost = self._calculate_cost(nearest_node, new_potential_node)

                 for near_node in near_nodes:
                     # Check collision for potential new parent connection
                     if not self._is_collision(near_node, new_potential_node, dynamic_preds, static_features):
                          cost_via_near = self._calculate_cost(near_node, new_potential_node)
                          if cost_via_near < min_new_cost:
                               min_cost_node = near_node
                               min_new_cost = cost_via_near

                 # Add the new node
                 new_node = Node(new_potential_node.x, new_potential_node.y, cost=min_new_cost, parent=min_cost_node)
                 self.nodes.append(new_node)
                 if USE_KDTREE: self.kdtree = KDTree([[n.x, n.y] for n in self.nodes]) # Rebuild KDTree


                 # ... (Rewiring logic remains the same, but uses updated collision check) ...
                 for near_node in near_nodes:
                     if near_node == min_cost_node: continue
                     cost_via_new = self._calculate_cost(new_node, near_node)
                     if cost_via_new < near_node.cost:
                          # Check collision for rewire connection
                          if not self._is_collision(new_node, near_node, dynamic_preds, static_features):
                               near_node.parent = new_node
                               near_node.cost = cost_via_new

                 # ... (Goal connection check logic remains the same, uses updated collision check) ...
                 dist_to_goal = self._distance(new_node, self.goal_node)
                 if dist_to_goal <= self.step_size:
                     if not self._is_collision(new_node, self.goal_node, dynamic_preds, static_features):
                          final_cost = self._calculate_cost(new_node, self.goal_node)
                          if best_goal_node is None or final_cost < best_goal_node.cost:
                               goal_connected_node = Node(self.goal_node.x, self.goal_node.y, cost=final_cost, parent=new_node)
                               best_goal_node = goal_connected_node
                               self.logger.debug("Found potential path to goal with cost %.2f", final_cost)


            # Log progress periodically
            if i > 0 and i % 500 == 0:
                 self.logger.debug("RRT* Iteration: %d/%d, Tree Size: %d", i, self.max_iter, len(self.nodes))

        # --- Path Reconstruction & Return ---
        # (Keep logic from previous version with updated logging)
        elapsed_time = time.time() - start_time
        if best_goal_node:
            path = self._reconstruct_path(best_goal_node)
            self.logger.info("RRT* SUCCESS: Path found (%d waypoints, cost %.2f) in %.2f seconds.", len(path), best_goal_node.cost, elapsed_time)
            return path
        else:
            # ... (find closest node logic) ...
             closest_node_to_goal = self._get_nearest_node((self.goal_node.x, self.goal_node.y))
             if closest_node_to_goal:
                  path = self._reconstruct_path(closest_node_to_goal)
                  self.logger.warning("RRT* PARTIAL: Direct goal not reached. Returning path to closest node (%d waypoints) after %.2f seconds.", len(path), elapsed_time)
                  return path
             else:
                  self.logger.error("RRT* FAILED: No path found after %d iterations in %.2f seconds.", self.max_iter, elapsed_time)
                  return None


# --- Federated Learning Client Placeholder ---
class FederatedLearningClient:
    def __init__(self, model_to_train, fl_server_url="http://placeholder.fl.server"):
        self.logger = logging.getLogger(__name__)
        self.model = model_to_train # e.g., behavior_learner.model
        self.server_url = fl_server_url
        self.logger.info("Federated Learning Client initialized for server: %s", self.server_url)
        # TODO: Add TFF specific initializations if using TFF libraries

    def check_for_update_request(self):
        """Simulates checking if the FL server requests participation."""
        self.logger.debug("Checking for FL training round participation...")
        # TODO: Implement communication with FL server (e.g., HTTP request)
        participate = random.random() < 0.1 # Simulate 10% chance participation
        if participate:
            self.logger.info("Selected for FL training round.")
            # Server would typically send model version, hyperparameters etc.
            return {"participate": True, "model_version": "v1.2", "epochs": 1}
        else:
            # self.logger.debug("Not selected for this FL round.")
            return {"participate": False}

    def run_local_training(self, config, local_data):
        """Simulates running local training based on server config."""
        self.logger.info("Running local FL training with config: %s", config)
        # TODO:
        # 1. Get local training data (e.g., recent state history from object_state_history)
        # 2. Preprocess data
        # 3. Train self.model using config['epochs'], local_data
        # 4. Calculate model updates (e.g., weight differences)
        if not local_data:
             self.logger.warning("No local data available for FL training.")
             return None

        # Example: Simulate generating dummy updates
        self.logger.info("Simulating local training... generating dummy updates.")
        time.sleep(2) # Simulate training time
        dummy_updates = {"layer1_grad": np.random.rand(10).tolist(), "samples": len(local_data)}
        return dummy_updates

    def send_updates(self, updates):
        """Simulates sending model updates back to the FL server."""
        self.logger.info("Sending FL updates to server (%d update keys)...", len(updates) if updates else 0)
        # TODO: Implement secure communication (e.g., HTTPS POST) to FL server
        success = random.random() < 0.95 # Simulate success
        if success:
             self.logger.info("Successfully sent FL updates.")
             return True
        else:
             self.logger.error("Simulated failure sending FL updates.")
             return False

    def receive_global_model(self):
        """Simulates receiving and applying the aggregated global model."""
        self.logger.info("Attempting to receive updated global model from FL server...")
        # TODO: Implement communication to fetch new global model weights
        use_new_model = random.random() < 0.8 # Simulate receiving an update
        if use_new_model:
             self.logger.info("Received new global model weights (simulated). Applying...")
             # In reality: model.set_weights(new_weights)
             time.sleep(0.5)
             self.logger.info("Local model updated with global weights.")
             return True
        else:
             self.logger.info("No new global model available from server.")
             return False

    def run_fl_round(self, local_data):
        """Orchestrates one round of federated learning participation."""
        round_info = self.check_for_update_request()
        if round_info and round_info.get("participate"):
            updates = self.run_local_training(round_info, local_data)
            if updates:
                if self.send_updates(updates):
                    # Optionally try to get global model immediately after contributing
                    self.receive_global_model()
        else:
            # Optionally check for new global model even if not training this round
            if random.random() < 0.2: # Check occasionally
                 self.receive_global_model()


# --- Main Robot Loop (Integrated) ---
def robot_loop():
    loop_logger = logging.getLogger("RobotLoop")
    loop_logger.info("--- Starting Robot Loop ---")

    # Initialization Block
    try:
        loop_logger.info("Initializing models and components...")
        behavior_learner = ObjectBehaviorLearner(state_dimension=KALMAN_STATE_DIM)
        neptune_client = NeptuneAPIClientPlaceholder() # Instantiate placeholder

        # ... (Calculate combiner dims) ...
        env_dim = 2; robot_dim = KALMAN_STATE_DIM; obstacle_dim = 1 # Example dims
        context_dim = env_dim + robot_dim + obstacle_dim
        transformer_output_dim = KALMAN_STATE_DIM
        combiner_output_dim = 2
        prediction_horizon = 10
        combiner = PredictionCombiner(transformer_output_dim, context_dim, combiner_output_dim, prediction_horizon)
        combiner.compile(optimizer='adam', loss='mse')

        # Pass neptune_client to navigator and planner
        navigator = RobotNavigator(behavior_learner, neptune_client)
        planner = PathPlannerRRTStar(behavior_learner, neptune_client, CONFIG_SPACE_BOUNDS)
        navigator.planner = planner # Make planner accessible to navigator for map_actions...

        # Initialize FL Client
        fl_client = FederatedLearningClient(behavior_learner.model)

        trackers = {}
        object_state_history = {}
        history_length = 10

        loop_logger.info("Initializing ROS2 node...")
        rclpy.init(args=None)
        environment_info_node = EnvironmentInfoNode()

        loop_logger.info("Opening video capture...")
        capture = cv2.VideoCapture(0)
        if not capture.isOpened(): raise IOError("Cannot open webcam")

        robot_id = 'robot_1'
        frame_count = 0
        last_fl_check_time = time.time()
        fl_check_interval = 60 # Seconds - check for FL rounds periodically

        loop_logger.info("Initialization complete.")

    except Exception as e:
        loop_logger.critical("Initialization failed: %s. Exiting.", e, exc_info=True)
        if 'capture' in locals() and isinstance(capture, cv2.VideoCapture) and capture.isOpened(): capture.release()
        if 'rclpy' in sys.modules and rclpy.ok(): rclpy.shutdown()
        sys.exit(1)

    # Main Loop Block
    try:
        while rclpy.ok():
            loop_start_time = time.time()
            # --- ROS2 ---
            rclpy.spin_once(environment_info_node, timeout_sec=0.01)

            # --- Perception ---
            ret, frame = capture.read()
            if not ret: # Handle camera read failure
                loop_logger.warning("Failed to read frame from camera. Skipping iteration.")
                time.sleep(0.1)
                continue
            frame_count += 1
            if frame_count % 100 == 0: loop_logger.debug("Processing frame %d", frame_count)

            # --- Main Processing Block ---
            try:
                detections = detect_objects(frame, object_detector)
                current_robot_state = get_robot_state()
                current_environment_info_comb = get_environment_info_for_combiner()
                current_obstacles_comb = process_lidar_data_for_obstacles(environment_info_global)

                processed_object_ids = set()
                object_predictions_transformer = {}
                object_actual_states = {}
                local_fl_data_batch = [] # Collect data for potential FL round

                # --- Tracking, Prediction, State Update Loop ---
                for detection in detections:
                     try:
                        object_id = detection.get('id')
                        measurement = detection.get('measurement')
                        if object_id is None or measurement is None: continue
                        processed_object_ids.add(object_id)

                        # Kalman Init/Update
                        if object_id not in trackers:
                            initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
                            trackers[object_id] = KalmanTracker(initial_state)
                            object_state_history[object_id] = deque(maxlen=history_length)
                        kalman_tracker = trackers[object_id]
                        predicted_state_kf = kalman_tracker.predict()
                        updated_state_kf = kalman_tracker.update(measurement)

                        if updated_state_kf is None: continue # Kalman update failed

                        current_state_flat = updated_state_kf.flatten()
                        object_actual_states[object_id] = current_state_flat
                        current_timestamp_iso = datetime.now(timezone.utc).isoformat()

                        # Update local cache AND persistent store (via client)
                        behavior_learner.update_local_cache(object_id, updated_state_kf, current_timestamp_iso)
                        neptune_client.upsert_object_state(object_id, current_state_flat, current_timestamp_iso) # Send update

                        # State History for Transformer & FL
                        object_state_history[object_id].append(current_state_flat)
                        if len(object_state_history[object_id]) == history_length:
                             local_fl_data_batch.append(list(object_state_history[object_id])) # Append sequence

                        # Transformer Prediction
                        if len(object_state_history[object_id]) >= history_length:
                             history_array = np.array(list(object_state_history[object_id])).reshape(1, history_length, KALMAN_STATE_DIM)
                             transformer_pred_seq = behavior_learner.predict(history_array)
                             if transformer_pred_seq is not None:
                                 current_transformer_pred = transformer_pred_seq[0, -1, :]
                                 object_predictions_transformer[object_id] = current_transformer_pred
                                 # Graph update is now handled by upsert_object_state
                             # else: Handle prediction failure? Maybe log.

                        # Optional Gemini Object ID (needs integration with update_object_semantics)
                        # ...

                     except (ValueError, Exception) as e_det: # Catch errors processing this detection
                          loop_logger.error("Error processing detection %s: %s", detection.get('id', 'N/A'), e_det, exc_info=True)

                # Remove Old Trackers
                # ... (logic remains same) ...
                lost_track_ids = set(trackers.keys()) - processed_object_ids
                for lost_id in lost_track_ids:
                     loop_logger.info("Object %s lost track.", lost_id)
                     if lost_id in trackers: del trackers[lost_id]
                     if lost_id in object_state_history: del object_state_history[lost_id]


                # --- Combiner Prediction ---
                # ... (logic remains same, using object_predictions_transformer) ...
                waypoints = None
                if object_predictions_transformer:
                    # ... (prepare inputs, call combiner, handle errors) ...
                    try:
                         # Simplistic averaging - improve this
                         if object_actual_states and object_predictions_transformer:
                              avg_transformer_pred = np.mean(np.array(list(object_predictions_transformer.values())), axis=0)
                              errors = [np.linalg.norm(object_actual_states[oid] - object_predictions_transformer[oid])
                                        for oid in object_predictions_transformer if oid in object_actual_states]
                              transformer_confidence = np.array([1.0 / (1.0 + np.mean(errors if errors else [1.0]))], dtype=np.float32)

                              combiner_input_list = [
                                 tf.constant(avg_transformer_pred.reshape(1, -1), dtype=tf.float32), # Needs correct shape
                                 tf.constant(transformer_confidence.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_environment_info_comb.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_robot_state.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_obstacles_comb.reshape(1, -1), dtype=tf.float32)
                              ]
                              # Ensure combiner input shapes match definition!
                              waypoints_tensor = combiner(combiner_input_list, training=False)
                              if waypoints_tensor is not None: waypoints = waypoints_tensor.numpy()[0]

                    except Exception as e_comb:
                         loop_logger.error("Error during prediction combination: %s", e_comb, exc_info=True)


                # --- Navigation Decision ---
                user_command = get_user_command()
                if user_command:
                    loop_logger.info("Processing user command: '%s'", user_command)
                    try:
                        # Pass current state info needed by planner/navigator
                        navigator.update_context(current_robot_state, current_environment_info_comb)
                        navigator.run_navigation(user_command) # This now uses planner via neptune client
                    except Exception as e_nav:
                        loop_logger.error("Error during command execution flow: %s", e_nav, exc_info=True)
                elif waypoints is not None:
                     # ... (Execute combiner path) ...
                     try:
                          smoothed_path = smooth_path(waypoints)
                          execute_path(smoothed_path)
                     except Exception as e_path:
                          loop_logger.error("Error executing combiner-generated path: %s", e_path, exc_info=True)

                else:
                    loop_logger.debug("Idle: No command or waypoints.")
                    time.sleep(0.1)

                # --- Federated Learning ---
                current_time = time.time()
                if current_time - last_fl_check_time > fl_check_interval:
                    loop_logger.info("Running periodic Federated Learning check...")
                    try:
                        # Pass recently collected data batch for potential training
                        fl_client.run_fl_round(local_fl_data_batch)
                        local_fl_data_batch = [] # Clear batch after use
                    except Exception as e_fl:
                        loop_logger.error("Error during Federated Learning round: %s", e_fl, exc_info=True)
                    last_fl_check_time = current_time

                # --- Remove GCS Graph Saving ---
                # The periodic save_graph_to_cloud_storage call is REMOVED.
                # Persistence is handled via neptune_client.upsert_object_state calls.

            except Exception as e_iter:
                 loop_logger.error("Unhandled error in main loop iteration: %s", e_iter, exc_info=True)
                 time.sleep(1)

            # --- Loop Timing/Debugging ---
            loop_duration = time.time() - loop_start_time
            loop_logger.debug("Loop iteration %d finished in %.3f seconds", frame_count, loop_duration)
            # Optional: Add delay if loop runs too fast
            # time.sleep(max(0, 0.05 - loop_duration)) # Example: target 20Hz


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
    logger.info("   Starting Robot Application (Neptune Arch)")
    logger.info("========================================")
    # Instantiate placeholder Neptune client here
    neptune_client = NeptuneAPIClientPlaceholder()
    try:
        # Pass neptune_client into robot_loop if needed, or access globally (less ideal)
        # Modifying robot_loop to accept it is cleaner but requires changing its signature.
        # For simplicity now, assume classes within robot_loop can access the global 'neptune_client'.
        # A better way is dependency injection.
        robot_loop() # Assuming robot_loop initializes components that use the global neptune_client
    except SystemExit as e:
         logger.warning("Application exited with code %s", e.code)
    except Exception as e:
        logger.critical("Application crashed at top level: %s", e, exc_info=True)
    finally:
        logger.info("========================================")
        logger.info("   Robot Application Finished           ")
        logger.info("========================================")
        logging.shutdown()
