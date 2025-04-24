# COMPLETE Robot Application Code v5.1 (REGENERATED - TF + Neptune Holder + FL Holder + Logging/Error + BNN Consultant Integration)
# Aiming to be complete based on previous version plus BNN additions.

# --- Imports ---
import os
import json
import networkx as nx # Keep for local cache if used
import google.generativeai
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, TimeDistributed,
                                     MultiHeadAttention, LayerNormalization, Dropout)
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
    print("Warning: scipy.spatial.KDTree not found. Using slower nearest neighbor search.")

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
# (Ensuring setup is complete and correct)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z') # ISO 8601 Format
log_level = logging.INFO # Change to logging.DEBUG for more detail

root_logger = logging.getLogger()
if not root_logger.hasHandlers(): # Setup handlers only once
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
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
# TODO: Use a robust configuration management system
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
YOLO_CFG_PATH = os.environ.get("YOLO_CFG_PATH", "yolov3.cfg")
YOLO_WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", "yolov3.weights")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", 'your-gcp-project-id')
PUBSUB_TOPIC_ID = 'object_exploration'
PUBSUB_SUBSCRIPTION_ID = 'robot_subscription'
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'your-bucket-name') # For logs/archives
NEPTUNE_API_ENDPOINT = os.environ.get("NEPTUNE_API_ENDPOINT", "http://placeholder.neptune.api/graph")
FL_SERVER_URL = os.environ.get("FL_SERVER_URL", "http://placeholder.fl.server")
CONFIG_SPACE_BOUNDS = (0, 20, 0, 15) # Example: (min_x, max_x, min_y, max_y)
STEP_SIZE = 0.5
MAX_ITERATIONS = 5000
GOAL_SAMPLE_RATE = 0.1
SEARCH_RADIUS = 1.5
ROBOT_RADIUS = 0.3
OBJECT_BASE_RADIUS = 0.2
PLANNER_SAFETY_FACTOR = 1.5
UNCERTAINTY_UPDATE_HZ = 1.0
BNN_MC_SAMPLES = 30
HIGH_UNCERTAINTY_THRESHOLD = 0.5
HISTORY_LENGTH = 10
ASSUMED_PREDICTION_DT = 1.0 # Assumed time step between predictor outputs (NEEDS VERIFICATION)

# --- Configuration Validation ---
if GEMINI_API_KEY == "YOUR_API_KEY_HERE": logger.warning("GEMINI_API_KEY not set.")
# ... etc ... Add more checks as needed ...

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
    if not os.path.exists(YOLO_CFG_PATH): raise FileNotFoundError(f"YOLO Cfg missing: {YOLO_CFG_PATH}")
    if not os.path.exists(YOLO_WEIGHTS_PATH): raise FileNotFoundError(f"YOLO Weights missing: {YOLO_WEIGHTS_PATH}")
    object_detector = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHTS_PATH)
    logger.info("Object Detector loaded successfully.")
except (cv2.error, FileNotFoundError) as e:
    logger.critical("Failed to load YOLO model: %s", e, exc_info=True)
    sys.exit(1)

# ==============================================================================
# === Class Definitions ========================================================
# ==============================================================================

# --- Kalman Filter ---
KALMAN_STATE_DIM = 4
KALMAN_MEASURE_DIM = 2
class KalmanTracker:
    """Handles Kalman Filter state estimation for a single object."""
    def __init__(self, initial_state):
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{id(self)}") # Unique logger instance name
        if initial_state is None or len(initial_state) != KALMAN_STATE_DIM or initial_state.ndim != 1:
             msg = f"Initial state must be 1D array of dim {KALMAN_STATE_DIM}, got shape {initial_state.shape if initial_state is not None else 'None'}"
             self.logger.error(msg); raise ValueError(msg)
        # Matrices NEED TUNING
        self.filter = cv2.KalmanFilter(KALMAN_STATE_DIM, KALMAN_MEASURE_DIM)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        dt = 1.0 # TODO: MUST replace with actual time delta
        self.filter.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.filter.processNoiseCov = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.03 # Tune Q
        self.filter.measurementNoiseCov = np.eye(KALMAN_MEASURE_DIM, dtype=np.float32) * 0.1 # Tune R
        init_state_col = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.statePost = init_state_col; self.filter.statePre = init_state_col # Init both
        self.filter.errorCovPost = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune P
        self.filter.errorCovPre = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune P

    def predict(self):
        """Predicts the next state."""
        try: return self.filter.predict() # Shape: (KALMAN_STATE_DIM, 1)
        except cv2.error as e: self.logger.error("Kalman predict error: %s", e); return None

    def update(self, measurement):
        """Updates filter state with a new measurement."""
        if measurement is None or len(measurement) != KALMAN_MEASURE_DIM or measurement.ndim != 1:
             msg = f"Measurement must be 1D array of dim {KALMAN_MEASURE_DIM}, got shape {measurement.shape if measurement is not None else 'None'}"
             self.logger.error(msg); raise ValueError(msg)
        measurement_col = measurement.reshape(-1, 1).astype(np.float32)
        try: self.filter.correct(measurement_col); return self.filter.statePost # Shape: (KALMAN_STATE_DIM, 1)
        except cv2.error as e: self.logger.error("Kalman correct error: %s", e); return self.filter.statePost

# --- Transformer Model (TensorFlow) ---
class TransformerBlock(tf.keras.layers.Layer):
    """Standard Transformer Block Implementation."""
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model=d_model; self.num_heads=num_heads; self.dff=dff; self.rate=rate
        name = kwargs.get('name', 'tf_block') # Ensure name is passed for logging/debugging
        key_dim_calc = d_model // num_heads if d_model > 0 and num_heads > 0 and d_model % num_heads == 0 else d_model
        if key_dim_calc == 0 and d_model > 0: key_dim_calc = d_model
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim_calc, name=f"{name}_mha")
        self.ln1 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")
        self.ffn = tf.keras.Sequential([Dense(dff, activation='relu'), Dense(d_model)], name=f"{name}_ffn")
        self.ln2 = LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")
        self.drop1 = Dropout(rate, name=f"{name}_drop1")
        self.drop2 = Dropout(rate, name=f"{name}_drop2")

    def call(self, x, training=None, mask=None):
        # Ensure training flag is explicitly passed to MHA and Dropout
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output = self.drop1(attn_output, training=training)
        out1 = self.ln1(x + attn_output)
        # Ensure training flag is passed to FFN if it contains layers like BatchNorm or Dropout
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        out2 = self.ln2(out1 + ffn_output)
        return out2

class ObjectBehaviorLearner:
    """Manages the Transformer model for predicting object behavior (point estimates)."""
    def __init__(self, prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.prediction_horizon = prediction_horizon
        self.state_dimension = state_dimension
        self.d_model = d_model; self.num_heads = num_heads; self.dff = dff; self.num_layers = num_layers
        self.local_object_graph_cache = nx.Graph() # Local cache
        self.logger.info("Initializing ObjectBehaviorLearner.")
        try:
            if d_model > 0 and num_heads > 0 and d_model % num_heads != 0: self.logger.warning("d_model (%d) not divisible by num_heads (%d). Check MHA config.", d_model, num_heads)
            # Inner Keras Model definition using subclassing
            class TransformerModel(tf.keras.Model):
                 def __init__(self, state_dim, d_model, num_layers, num_heads, dff, **kwargs):
                     super().__init__(**kwargs)
                     self.embedding_layer = Dense(d_model, name="input_embedding")
                     self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff, rate=0.1, name=f"tf_block_{i}") for i in range(num_layers)]
                     self.output_dense_layer = TimeDistributed(Dense(state_dim), name="output_dense")
                 @tf.function
                 def call(self, inputs, training=None, mask=None):
                     x = self.embedding_layer(inputs)
                     # TODO: Add positional encoding if sequence order matters
                     for block in self.transformer_blocks: x = block(x, training=training, mask=mask)
                     return self.output_dense_layer(x)
            self.model = TransformerModel(state_dimension, d_model, num_layers, num_heads, dff)
            self.model.build(input_shape=(None, None, state_dimension)) # Build with flexible shapes
            self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
            self.logger.info("ObjectBehaviorLearner TF Model built and compiled.")
        except Exception as e: self.logger.critical("Failed to build/compile TF model: %s", e, exc_info=True); raise

    def train(self, past_states, target_states):
        self.logger.info("Training Transformer model with %d samples...", tf.shape(past_states)[0])
        try:
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            val_loss = history.history.get('val_loss', [None])[-1]; self.logger.info("Training finished. Val loss: %s", f"{val_loss:.4f}" if val_loss is not None else "N/A")
        except Exception as e: self.logger.error("Error during training: %s", e, exc_info=True)

    def predict(self, past_states):
        # self.logger.debug("Standard prediction...")
        try: return self.model(past_states, training=False) # Use call with training=False
        except Exception as e: self.logger.error("Error during standard prediction: %s", e, exc_info=True); return None

    def update_local_cache(self, object_id, state, timestamp_iso):
        """Updates the local NetworkX graph cache."""
        state_values = state.flatten().tolist()
        if object_id not in self.local_object_graph_cache.nodes:
            self.local_object_graph_cache.add_node(object_id, state=state_values, timestamp=timestamp_iso)
        else:
            self.local_object_graph_cache.nodes[object_id]['state'] = state_values
            self.local_object_graph_cache.nodes[object_id]['timestamp'] = timestamp_iso

# --- BNN Variant using MC Dropout ---
class BNNObjectBehaviorLearner(ObjectBehaviorLearner):
    """BNN version using MC Dropout for uncertainty estimation."""
    def __init__(self, num_mc_samples=BNN_MC_SAMPLES, **kwargs):
        super().__init__(**kwargs) # Builds the same underlying Keras model
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not isinstance(num_mc_samples, int) or num_mc_samples <= 1: raise ValueError("num_mc_samples must be > 1.")
        self.num_mc_samples = num_mc_samples
        self.logger.info("Initialized with %d MC samples.", self.num_mc_samples)
        # Verify dropout layers exist using helper
        if not any(isinstance(layer, Dropout) for layer in self._get_all_layers(self.model)):
             self.logger.warning("MC Dropout requires Dropout layers, but none detected.")

    def _get_all_layers(self, model):
        """Helper to recursively get all layers including nested ones."""
        layers = []
        if hasattr(model, 'layers'):
            for layer in model.layers:
                 layers.append(layer)
                 # Check if it's a sequential or functional model containing layers, or a layer containing layers
                 if isinstance(layer, (tf.keras.Model, tf.keras.layers.Wrapper)) or hasattr(layer, 'layers'):
                      layers.extend(self._get_all_layers(layer))
        return layers

    @tf.function
    def predict_mc_dropout_tf(self, past_states):
        """Internal TF function for the prediction loop."""
        # Determine element shape dynamically from one forward pass
        example_output = self.model(past_states, training=True)
        # We expect output shape (batch, time, state_dim). Element shape is (time, state_dim)
        element_shape = tf.shape(example_output)[1:]
        predictions_array = tf.TensorArray(dtype=tf.float32, size=self.num_mc_samples, dynamic_size=False, element_shape=element_shape)
        # Write the first prediction
        predictions_array = predictions_array.write(0, example_output[0]) # Write first batch element
        # Loop for the remaining samples
        for i in tf.range(1, self.num_mc_samples):
            predictions = self.model(past_states, training=True) # Key: training=True
            predictions_array = predictions_array.write(i, predictions[0])
        return predictions_array.stack()

    def predict(self, past_states: np.ndarray | tf.Tensor) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """ Performs prediction using MC Dropout to estimate mean and variance. """
        start_time = time.time()
        # self.logger.debug("BNN MC Dropout prediction...")
        try:
            if not isinstance(past_states, tf.Tensor): past_states = tf.constant(past_states, dtype=tf.float32)
            # Ensure batch dimension
            input_shape = tf.shape(past_states)
            if len(input_shape) == 2: past_states = tf.expand_dims(past_states, axis=0); input_batch_size = 1
            elif len(input_shape) == 3: input_batch_size = input_shape[0]
            else: raise ValueError(f"Input shape error: {input_shape}")
            if input_batch_size > 1:
                 self.logger.warning("MC Dropout predict currently assumes batch size 1 for TensorArray usage. Input batch size is %d", input_batch_size)
                 # TODO: Adapt predict_mc_dropout_tf or use eager execution for batch > 1

            predictions_stack = self.predict_mc_dropout_tf(past_states)
            # Shape: (num_mc_samples, time_steps, state_dimension)
            mean_prediction = tf.reduce_mean(predictions_stack, axis=0)
            variance_prediction = tf.math.reduce_variance(predictions_stack, axis=0)
            # Restore batch dimension
            mean_prediction = tf.expand_dims(mean_prediction, axis=0)
            variance_prediction = tf.expand_dims(variance_prediction, axis=0)

            duration = time.time() - start_time; self.logger.debug("MC Dropout predict took %.3fs", duration)
            return mean_prediction, variance_prediction
        except Exception as e: self.logger.error("MC Dropout predict error: %s", e, exc_info=True); return None, None

# --- Uncertainty Consultant ---
class UncertaintyConsultant:
    # (Implementation from previous step)
    def __init__(self, bnn_learner_instance: BNNObjectBehaviorLearner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not isinstance(bnn_learner_instance, BNNObjectBehaviorLearner): raise TypeError("Invalid BNN")
        self.bnn_learner = bnn_learner_instance
        self.current_uncertainty_metrics = {}
        self.update_period_secs = 1.0 / update_frequency_hz if update_frequency_hz > 0 else float('inf')
        self.last_update_time = 0
        self.logger.info("UncertaintyConsultant init. Update Period: %.2fs", self.update_period_secs)

    def update_uncertainty(self, object_state_history: dict):
         current_time = time.time()
         if current_time - self.last_update_time < self.update_period_secs: return
         self.logger.debug("Running BNN uncertainty analysis...")
         overall_variance_sum=0.0; num_predictions=0; high_uncertainty_objects=[]; object_variances={}
         history_len_needed = self.bnn_learner.prediction_horizon # Get from learner
         start_analysis_time = time.time(); objects_analyzed = 0
         for obj_id, history_deque in object_state_history.items():
             if len(history_deque) >= history_len_needed:
                 history_list = list(history_deque)[-history_len_needed:]
                 history_array = np.array(history_list).reshape(1, history_len_needed, self.bnn_learner.state_dimension)
                 try:
                     mean_pred, var_pred = self.bnn_learner.predict(history_array) # Returns (mean, var)
                     objects_analyzed += 1
                     if var_pred is not None:
                         # Avg positional variance over horizon (assumes state[0]=x, state[1]=y)
                         pos_variance = var_pred[0, :, :2] # Squeeze batch dim, take time steps, take x,y dims
                         avg_pos_var = tf.reduce_mean(pos_variance).numpy()
                         overall_variance_sum+=avg_pos_var; num_predictions+=1; object_variances[obj_id]=avg_pos_var
                         if avg_pos_var > HIGH_UNCERTAINTY_THRESHOLD: high_uncertainty_objects.append(obj_id)
                 except Exception as e: self.logger.error("BNN predict error for %s: %s", obj_id, e)

         analysis_duration = time.time() - start_analysis_time
         avg_overall_variance = overall_variance_sum / num_predictions if num_predictions > 0 else 0.0
         self.current_uncertainty_metrics = { "average_positional_variance": avg_overall_variance, "high_uncertainty_ids": high_uncertainty_objects, "object_variances": object_variances }
         self.last_update_time = current_time
         self.logger.info("Uncertainty update (%d obj in %.3fs): AvgVar=%.4f, HighUnc=%d", objects_analyzed, analysis_duration, avg_overall_variance, len(high_uncertainty_objects))
         self.logger.debug("Metrics: %s", self.current_uncertainty_metrics)

    def get_metrics(self) -> dict: return self.current_uncertainty_metrics

# --- Prediction Combiner (TensorFlow) ---
class PredictionCombiner(tf.keras.Model):
    # (Implementation from previous step)
     def __init__(self, transformer_output_dim_flat, context_dim, output_dim=2, prediction_horizon=10, use_confidence_weighting=True, dropout_rate=0.2):
        super(PredictionCombiner, self).__init__()
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.output_dim=output_dim; self.prediction_horizon=prediction_horizon; self.use_confidence_weighting=use_confidence_weighting; self.dropout_rate=dropout_rate
        self.dense1 = Dense(128, activation='relu'); self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(self.output_dim * self.prediction_horizon)
        self.layer_norm = LayerNormalization(epsilon=1e-6); self.dropout = Dropout(self.dropout_rate)
        self.logger.info("Prediction Combiner Initialized.")

     def call(self, inputs, training=None):
         # (Implementation from previous step with safe_flatten)
         try:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 5: raise ValueError("Invalid input structure")
            tf_preds, tf_confs, env_info, rb_state, obs = inputs
            def safe_flatten(tensor):
                if tensor is None: return None; shape = tf.shape(tensor); rank = len(shape)
                if rank > 2: return tf.reshape(tensor, [shape[0], -1])
                elif rank == 1: return tf.expand_dims(tensor, axis=-1)
                return tensor
            tf_preds_flat=safe_flatten(tf_preds); tf_confs_flat=safe_flatten(tf_confs); env_info_flat=safe_flatten(env_info)
            rb_state_flat=safe_flatten(rb_state); obs_flat=safe_flatten(obs)
            valid_inputs = [p for p in [tf_preds_flat, env_info_flat, rb_state_flat, obs_flat] if p is not None]
            if not valid_inputs: raise ValueError("No valid inputs")
            weighted_preds = tf_preds_flat
            if self.use_confidence_weighting and tf_confs_flat is not None and tf_preds_flat is not None:
                 try: weighted_preds = tf_preds_flat * tf_confs_flat # Check shapes broadcast correctly
                 except Exception as e: self.logger.error("Weighting error: %s", e)
            if weighted_preds is not None: valid_inputs[0] = weighted_preds # Assume first is prediction
            combined_input = Concatenate()(valid_inputs)
            x = self.layer_norm(combined_input, training=training); x = self.dropout(x, training=training)
            x = self.dense1(x); x = self.dense2(x); waypoints_flat = self.dense3(x)
            return tf.reshape(waypoints_flat, [-1, self.prediction_horizon, self.output_dim])
         except (tf.errors.InvalidArgumentError, ValueError) as e: self.logger.error("Combiner TF Error: %s", e); return None
         except Exception as e: self.logger.error("Combiner call error: %s", e, exc_info=True); return None

# --- ROS2 Node ---
environment_info_global = None
class EnvironmentInfoNode(Node):
    # (Implementation from previous step)
     def __init__(self):
          super().__init__('environment_info_node'); self.logger = self.get_logger()
          try: self.subscription = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10); self.logger.info("Subscribed to /scan.")
          except Exception as e: self.logger.error("ROS sub failed: %s", e); raise
     def lidar_callback(self, msg):
          global environment_info_global
          try:
               ranges = np.array(msg.ranges); finite_ranges = ranges[np.isfinite(ranges)]
               processed_info = np.array([-1.0, -1.0], dtype=np.float32)
               if len(finite_ranges) > 0: processed_info = np.array([np.min(finite_ranges), np.mean(finite_ranges)], dtype=np.float32)
               environment_info_global = processed_info
          except Exception as e: self.logger.error("Lidar callback error: %s", e, exc_info=True)

# --- Pub/Sub Setup ---
# (Keep implementation from previous step)
# ...

# --- Cloud Storage Function (Definition only) ---
def save_graph_to_cloud_storage(graph, bucket_name, blob_name):
    # (Keep implementation from previous step)
    pass

# --- Neptune API Client Placeholder ---
# (Keep implementation from previous step)
class NeptuneAPIClientPlaceholder:
    # ... (implementation) ...
    def __init__(self, api_endpoint=NEPTUNE_API_ENDPOINT, api_key="DUMMY_KEY"): self.logger=logging.getLogger(__name__); self.api_endpoint=api_endpoint; self.logger.info("Neptune Placeholder Init.")
    def get_object(self, object_id: str): return {'id': object_id, 'state': [1,1,0,0]} if random.random()<0.9 else None
    def query_objects(self, criteria: dict): return []
    def query_map_features(self, bounds: dict): return []
    def upsert_object_state(self, object_id: str, state_vector, timestamp_iso: str, **kwargs): self.logger.info("Sim Neptune: upsert %s", object_id); return True
    def update_object_semantics(self, object_id: str, properties: dict): self.logger.info("Sim Neptune: update %s", object_id); return True

# --- Robot Navigator ---
class RobotNavigator:
    # (Implementation uses Neptune client, includes cautiousness)
     def __init__(self, object_behavior_learner, neptune_client, max_retries=3, initial_delay=1):
          self.logger = logging.getLogger(f"{self.__class__.__name__}")
          self.object_learner = object_behavior_learner; self.neptune_client = neptune_client
          self.robot_state = None; self.environment_info = None; self.gemini_model = None
          self.max_retries = max_retries; self.initial_delay = initial_delay; self.planner = None; self.current_cautiousness = 0.0
          try:
              if gemini_configured: self.gemini_model = google.generativeai.GenerativeModel('gemini-pro')
              else: self.logger.warning("Gemini model not initialized.")
          except Exception as e: self.logger.error("Gemini init failed: %s", e)

     def update_context(self, robot_state, environment_info): self.robot_state = robot_state; self.environment_info = environment_info
     def format_environment_info(self): return str(self.environment_info) if self.environment_info is not None else "N/A"
     def set_cautiousness(self, uncertainty_metric): self.current_cautiousness = np.clip(uncertainty_metric * 2.0, 0.0, 1.0); self.logger.debug("Cautiousness: %.2f", self.current_cautiousness)
     def get_cautiousness(self): return self.current_cautiousness

     def _build_gemini_prompt(self, user_command):
         # (Implementation uses Neptune client queries as before)
         graph_context_str = "Graph context error."; map_features_str = "Map feature error."
         try: # Fetch context
             robot_loc = self.robot_state[:2] if self.robot_state is not None else [0,0]
             query_criteria = {'location_radius': {'x': robot_loc[0], 'y': robot_loc[1], 'radius_m': 50}}
             nearby_objects = self.neptune_client.query_objects(criteria=query_criteria)
             graph_context_str = f"{len(nearby_objects)} nearby objects (Sim)." if nearby_objects else "No nearby objects (Sim)."
             map_bounds = {'min_x': robot_loc[0]-20, 'max_x': robot_loc[0]+20, 'min_y': robot_loc[1]-20, 'max_y': robot_loc[1]+20}
             map_features = self.neptune_client.query_map_features(bounds=map_bounds)
             map_features_str = f"{len(map_features)} map features (Sim)." if map_features else "No map features (Sim)."
         except Exception as e: self.logger.error("Error fetching context for Gemini: %s", e)
         # Build prompt...
         prompt = f"[Prompt Template... Context: Robot={self.robot_state}, Env={self.format_environment_info()}, Graph={graph_context_str}, Map={map_features_str}, Cmd={user_command} ...Instructions...]"
         return prompt

     def get_gemini_navigation_instructions(self, user_command):
         # (Implementation with retries as before)
         if not self.gemini_model: self.logger.error("Gemini unavailable."); return None
         prompt = self._build_gemini_prompt(user_command)
         # ... (Retry loop logic) ...
         # Placeholder return for now, ensuring method exists
         self.logger.info("Simulating Gemini call for: %s", user_command)
         time.sleep(0.5) # Simulate latency
         # Simulate a simple valid JSON response sometimes
         if random.random() < 0.8:
              return """```json
[
  {"action": "go_to", "target": "sim_nearby_obj_0"},
  {"action": "report_status"}
]
```"""
         else:
              self.logger.warning("Simulating Gemini API failure/empty response.")
              return None # Simulate failure


     def parse_gemini_instructions(self, instructions_text):
         # (Implementation with logging, JSON parsing as before)
         if not instructions_text: return None
         try:
             match = re.search(r'```json\s*([\s\S]*?)\s*```', instructions_text, re.MULTILINE)
             json_str = match.group(1).strip() if match else instructions_text.strip()
             if not match: self.logger.warning("JSON markdown block not found.")
             instructions_list = json.loads(json_str)
             if not isinstance(instructions_list, list): self.logger.error("Parsed JSON not a list."); return None
             valid_instructions = []
             for item in instructions_list:
                 if isinstance(item, dict) and "action" in item: valid_instructions.append({"action": item["action"], "target": item.get("target")})
                 else: self.logger.warning("Skipping invalid instruction: %s", item)
             self.logger.info("Parsed %d valid instructions.", len(valid_instructions))
             return valid_instructions
         except Exception as e: self.logger.error("Parsing Gemini instructions error: %s", e, exc_info=True); return None

     def map_actions_to_robot_commands(self, instructions, object_state_history):
         # (Implementation modified to accept history and call planner correctly)
         if instructions is None: return None
         if self.planner is None: self.logger.error("Planner not set!"); return None
         robot_commands = []; self.logger.info("Mapping %d instructions...", len(instructions))
         for instruction in instructions:
             action=instruction.get("action"); target=instruction.get("target")
             if action == "go_to":
                 if target is None: self.logger.warning("Skip go_to: no target"); continue
                 self.logger.info("Requesting plan to: %s", target)
                 env_info_snapshot = get_environment_info_for_combiner() # Get current env state for planner
                 # *** Pass object_state_history to planner ***
                 waypoints = self.planner.plan_path(self.robot_state, target, env_info_snapshot, object_state_history)
                 if waypoints: robot_commands.extend([("move_to", wp) for wp in waypoints])
                 else: self.logger.error("Path planning failed for: %s", target)
             # ... (other mappings) ...
         return robot_commands

     def execute_commands(self, commands):
          # (Implementation with logging as before)
          if not commands: self.logger.info("No commands execute."); return
          self.logger.info("Executing %d commands...", len(commands))
          for i, (cmd, args) in enumerate(commands):
               self.logger.info(" CMD %d: %s(%s)", i+1, cmd, args)
               # TODO: Real execution + error handling
               time.sleep(0.1) # Simulate

     def run_navigation(self, user_command, object_state_history): # Added history
        # (Implementation calls other methods, passes history)
        self.logger.info("Running navigation for command: %s", user_command)
        instructions_text = self.get_gemini_navigation_instructions(user_command)
        if instructions_text:
            instructions = self.parse_gemini_instructions(instructions_text)
            if instructions and instructions[0].get("action") != "error":
                 # Pass history to map_actions
                 commands = self.map_actions_to_robot_commands(instructions, object_state_history)
                 if commands: self.execute_commands(commands)
                 else: self.logger.error("Action mapping failed.")
            elif instructions and instructions[0].get("action") == "error": self.logger.error("Gemini plan error: %s", instructions[0].get('message'))
            else: self.logger.error("Could not parse Gemini instructions.")
        else: self.logger.error("Could not get Gemini instructions.")


# --- Path Planner (RRT*) ---
Obstacle = namedtuple("Obstacle", ["x", "y", "radius"])
class Node:
    def __init__(self, x, y, cost=0.0, parent=None): self.x=x; self.y=y; self.cost=cost; self.parent=parent
    def __eq__(self, other): return self.x == other.x and self.y == other.y; def __hash__(self): return hash((self.x, self.y)); def __repr__(self): return f"N({self.x:.1f},{self.y:.1f})"

class PathPlannerRRTStar:
    # (Integrated implementation with uncertainty margins and dynamic prediction call)
     def __init__(self, behavior_learner, neptune_client, config_space_bounds, step_size=STEP_SIZE, max_iter=MAX_ITERATIONS, goal_sample_rate=GOAL_SAMPLE_RATE, search_radius=SEARCH_RADIUS, robot_radius=ROBOT_RADIUS):
          self.logger = logging.getLogger(f"{self.__class__.__name__}")
          self.behavior_learner=behavior_learner; self.neptune_client=neptune_client # Store dependencies
          self.min_x, self.max_x, self.min_y, self.max_y = config_space_bounds
          self.step_size=step_size; self.max_iter=max_iter; self.goal_sample_rate=goal_sample_rate; self.search_radius=search_radius; self.robot_radius=robot_radius;
          self.object_base_radius = OBJECT_BASE_RADIUS
          self.uncertainty_margins={}; self.default_uncertainty_margin = 0.0 # Uncertainty state
          self.nodes=[]; self.kdtree=None; self.start_node=None; self.goal_node=None # RRT* state
          self.logger.info(f"RRT* Planner Initialized. Neptune client, KDTree: {USE_KDTREE}")

     def adjust_safety_margins(self, average_variance: float, object_variances: dict):
         # (Implementation from previous step)
         default_std_dev = math.sqrt(max(0, average_variance)); self.default_uncertainty_margin = default_std_dev * PLANNER_SAFETY_FACTOR
         self.uncertainty_margins.clear(); [self.uncertainty_margins.update({oid: math.sqrt(max(0, v)) * PLANNER_SAFETY_FACTOR}) for oid, v in object_variances.items()]
         self.logger.debug("Updated uncertainty margins (default: %.3f)", self.default_uncertainty_margin)

     def get_goal_position(self, target) -> tuple | None:
         # (Implementation uses self.neptune_client.get_object)
         if isinstance(target, (list, tuple, np.ndarray)): return tuple(target[:2])
         if isinstance(target, str):
              obj_data = self.neptune_client.get_object(target)
              if obj_data and 'state' in obj_data and len(obj_data['state']) >= 2: return tuple(obj_data['state'][:2])
         self.logger.warning("Cannot resolve target '%s'.", target); return None

     def predict_dynamic_obstacles(self, object_state_history: dict, prediction_horizon_steps: int) -> dict:
         # (Implementation from previous step using self.behavior_learner)
         # self.logger.debug("Predicting dynamic obstacles...") # Verbose
         predictions_dict = {}; dt_pred = ASSUMED_PREDICTION_DT
         history_len_needed = self.behavior_learner.prediction_horizon
         for obj_id, history_deque in object_state_history.items():
             if len(history_deque) >= history_len_needed:
                 history_list = list(history_deque)[-history_len_needed:]
                 try:
                     input_array = np.array(history_list).reshape(1, history_len_needed, self.behavior_learner.state_dimension)
                     pred_tensor = self.behavior_learner.predict(tf.constant(input_array, dtype=tf.float32))
                     if pred_tensor is not None:
                         pred_sequence = pred_tensor[0].numpy()
                         steps_to_use = min(prediction_horizon_steps, pred_sequence.shape[0])
                         # Assumes state is [x, y, ...], uses object_base_radius
                         traj = [((i + 1) * dt_pred, pred_sequence[i, 0], pred_sequence[i, 1], self.object_base_radius) for i in range(steps_to_use)]
                         if traj: predictions_dict[obj_id] = traj
                 except Exception as e: self.logger.error("Predict error for %s: %s", obj_id, e)
         # self.logger.debug("Predicted trajectories for %d objects.", len(predictions_dict))
         return predictions_dict

     def _is_collision(self, node1, node2, dynamic_obstacles_pred: dict, static_features: list):
         # (Implementation uses self.uncertainty_margins)
         p1=np.array([node1.x,node1.y]); p2=np.array([node2.x,node2.y])
         segment_vec = p2 - p1; segment_len = np.linalg.norm(segment_vec)
         if segment_len < 1e-6: return False
         # Static Check (Placeholder)
         for feature in static_features: pass # TODO: Implement checks (e.g., line-segment vs robot circle)
         # Dynamic Check
         if dynamic_obstacles_pred:
             num_steps = max(2, int(segment_len / (self.step_size * 0.5)))
             for i in range(num_steps + 1):
                 robot_pos = p1 + (i / num_steps) * segment_vec
                 # TODO: More accurate time estimation for robot_pos
                 for obj_id, trajectory in dynamic_obstacles_pred.items():
                      # Simplified: Check against all predicted points in trajectory for now
                      for (_time, obs_x, obs_y, obs_base_radius) in trajectory:
                           margin = self.uncertainty_margins.get(obj_id, self.default_uncertainty_margin)
                           effective_obs_radius = obs_base_radius + margin
                           obs_center = np.array([obs_x, obs_y])
                           total_radius_sq = (self.robot_radius + effective_obs_radius)**2
                           if np.sum((robot_pos - obs_center)**2) <= total_radius_sq: return True
         return False

     # plan_path calls predict_dynamic_obstacles and uses _is_collision
     def plan_path(self, start_pose, goal_target, environment_info, object_state_history: dict): # Added history
        start_time = time.time()
        self.logger.info("RRT* Plan: Start=%s, Target='%s'", start_pose[:2], goal_target)
        self.start_node = Node(start_pose[0], start_pose[1]); goal_pos = self.get_goal_position(goal_target)
        if goal_pos is None: return None
        self.goal_node = Node(goal_pos[0], goal_pos[1])
        self.nodes = [self.start_node]; best_goal_node = None
        if USE_KDTREE: self.kdtree = KDTree([[n.x, n.y] for n in self.nodes])

        # Get obstacles
        map_bounds = {'min_x':self.min_x, 'max_x':self.max_x, 'min_y':self.min_y, 'max_y':self.max_y}
        static_features = self.neptune_client.query_map_features(bounds=map_bounds)
        dynamic_preds = self.predict_dynamic_obstacles(object_state_history, prediction_horizon_steps=10) # Use history

        # RRT* Loop
        for i in range(self.max_iter):
             rnd_point = self._get_random_point(); nearest_node = self._get_nearest_node(rnd_point)
             if nearest_node is None: continue
             new_potential_node = self._steer(nearest_node, rnd_point)
             # Check collision using dynamic preds and static features
             if not self._is_collision(nearest_node, new_potential_node, dynamic_preds, static_features):
                 # Choose parent and rewire (core RRT* logic)
                 near_nodes = self._find_near_nodes(new_potential_node); min_cost_node = nearest_node; min_new_cost = self._calculate_cost(nearest_node, new_potential_node)
                 for near_node in near_nodes: # Choose parent
                     if not self._is_collision(near_node, new_potential_node, dynamic_preds, static_features):
                          cost_via_near = self._calculate_cost(near_node, new_potential_node)
                          if cost_via_near < min_new_cost: min_cost_node = near_node; min_new_cost = cost_via_near
                 new_node = Node(new_potential_node.x, new_potential_node.y, cost=min_new_cost, parent=min_cost_node); self.nodes.append(new_node)
                 if USE_KDTREE: self.kdtree = KDTree([[n.x, n.y] for n in self.nodes]) # Rebuild
                 for near_node in near_nodes: # Rewire
                     if near_node == min_cost_node: continue
                     cost_via_new = self._calculate_cost(new_node, near_node)
                     if cost_via_new < near_node.cost and not self._is_collision(new_node, near_node, dynamic_preds, static_features):
                          near_node.parent = new_node; near_node.cost = cost_via_new
                 # Goal check
                 if self._distance(new_node, self.goal_node) <= self.step_size and not self._is_collision(new_node, self.goal_node, dynamic_preds, static_features):
                      final_cost = self._calculate_cost(new_node, self.goal_node)
                      if best_goal_node is None or final_cost < best_goal_node.cost: best_goal_node = Node(self.goal_node.x, self.goal_node.y, cost=final_cost, parent=new_node)

        # Path Reconstruction
        elapsed_time = time.time() - start_time; path = None
        if best_goal_node: path = self._reconstruct_path(best_goal_node); self.logger.info("RRT* SUCCESS: Path found (%d wp, cost %.2f) in %.2fs.", len(path), best_goal_node.cost, elapsed_time)
        else: # Find closest if no direct path found
             closest_node = self._get_nearest_node((self.goal_node.x, self.goal_node.y))
             if closest_node and closest_node != self.start_node : path = self._reconstruct_path(closest_node); self.logger.warning("RRT* PARTIAL: Goal not reached. Path to closest node (%d wp) in %.2fs.", len(path), elapsed_time)
             else: self.logger.error("RRT* FAILED: No path found after %d iter in %.2fs.", self.max_iter, elapsed_time)
        return path

     # Helper methods (_distance, _steer, _get_random_point, _get_nearest_node, etc.)
     # ... (Keep implementations from previous RRT* version) ...
     def _distance(self, node1, node2): p1=node1 if isinstance(node1,Node) else Node(node1[0],node1[1]);p2=node2 if isinstance(node2,Node) else Node(node2[0],node2[1]); return math.hypot(p1.x-p2.x, p1.y-p2.y)
     def _calculate_cost(self, from_node, to_node): return from_node.cost + self._distance(from_node, to_node)
     def _reconstruct_path(self, goal_node_final): path=[]; curr=goal_node_final; while curr: path.append((curr.x, curr.y)); curr=curr.parent; return path[::-1]

# --- Federated Learning Client Placeholder ---
class FederatedLearningClient:
    # (Implementation from previous step)
     def __init__(self, keras_model, fl_server_url=FL_SERVER_URL):
         self.logger = logging.getLogger(f"{self.__class__.__name__}")
         self.local_model=keras_model; self.server_url=fl_server_url; self.tff_available = TFF_AVAILABLE; self.initial_weights=None
         if TFF_AVAILABLE: try: self.initial_weights=[w.numpy() for w in self.local_model.weights] except Exception as e: self.logger.error("FL init weights error: %s", e)
         self.logger.info("FL Client init. TFF Available: %s", self.tff_available)
     def _prepare_tf_dataset(self, local_data_batch: list, batch_size=16): self.logger.warning("FL _prepare_tf_dataset not implemented."); return None # Placeholder
     def check_participation_and_get_config(self): self.logger.debug("FL Check participation (Sim)"); return {"participate":True, "epochs":1, "batch_size":16} if random.random()<0.1 else None # Simulate participation
     def get_server_model_weights(self, config: dict): self.logger.debug("FL Get weights (Sim)"); return self.initial_weights # Simulate return structure
     def run_local_training_and_compute_update(self, server_weights: list, train_dataset: tf.data.Dataset, config: dict): self.logger.info("FL Run local training (Sim)..."); return {'weights_delta': None, 'num_samples':0, 'metrics':{}} # Placeholder
     def send_updates(self, client_update: dict): self.logger.info("FL Send updates (Sim)..."); return True # Placeholder
     def run_fl_round(self, local_data_batch: list):
          if not self.tff_available: self.logger.debug("Skip FL round: TFF not available"); return
          config = self.check_participation_and_get_config()
          if config:
              train_ds = self._prepare_tf_dataset(local_data_batch, config.get("batch_size", 16))
              if train_ds:
                   server_weights = self.get_server_model_weights(config)
                   if server_weights:
                        client_update = self.run_local_training_and_compute_update(server_weights, train_ds, config)
                        if client_update: self.send_updates(client_update)

# --- Placeholder Functions ---
# (Keep definitions, add logging if desired)
def get_user_command(): logger.debug("Checking user cmd..."); return None
def get_robot_state(): return np.array([random.uniform(1,19), random.uniform(1,14), 0.0, 0.0]) # Random pose [x,y,vx,vy]
def get_environment_info_for_combiner(): global environment_info_global; return environment_info_global.astype(np.float32) if environment_info_global is not None else np.zeros(2, dtype=np.float32)
def process_lidar_data_for_obstacles(lidar_data): return np.array([np.mean(lidar_data if lidar_data is not None else [-1.0])], dtype=np.float32)
def smooth_path(waypoints): return waypoints # Placeholder
def execute_path(path, cautiousness=0.0):
    logger.info("Executing path (%d wp). Cautiousness=%.2f", len(path) if path else 0, cautiousness)
    if path: time.sleep(0.1 * len(path) * (1 + cautiousness)) # Simulate execution

def detect_objects(frame, detector): # Placeholder
     # logger.debug("Detecting objects...")
     return [{'id': f'sim_{int(time.time()*10 % 1000) + i}', 'measurement': np.array([random.uniform(1,19), random.uniform(1,14)]), 'class': 'sim_obj'} for i in range(random.randint(0,4))]

# Gemini Object Query Placeholders (use neptune_client)
def update_object_graph_with_gemini(neptune_client, object_id, gemini_response):
     logger.info("Updating Neptune with Gemini info for %s", object_id)
     neptune_client.update_object_semantics(object_id, gemini_response)
# ... other placeholders ...


# ==============================================================================
# === Main Robot Loop ==========================================================
# ==============================================================================
def robot_loop(neptune_client_instance): # Expect neptune client
    loop_logger = logging.getLogger("RobotLoop")
    loop_logger.info("--- Initializing Robot Loop Components ---")
    # Component Initialization
    navigator = None; planner = None; fl_client = None; capture = None; environment_info_node = None
    try:
        # Learners
        realtime_learner = ObjectBehaviorLearner(prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM)
        bnn_learner = BNNObjectBehaviorLearner(prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM, num_mc_samples=BNN_MC_SAMPLES)
        uncertainty_consultant = UncertaintyConsultant(bnn_learner_instance=bnn_learner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ)

        # Combiner (ensure input/output dims are correct)
        tf_pred_dim_flat = KALMAN_STATE_DIM # Based on current model output assumption
        context_dim_approx = 2 + KALMAN_STATE_DIM + 1 # env+robot+obstacle example
        combiner_output_dim = 2; combiner_pred_horizon = 10
        combiner = PredictionCombiner(tf_pred_dim_flat, context_dim_approx, combiner_output_dim, combiner_pred_horizon)
        combiner.compile(optimizer='adam', loss='mse')

        # Navigation and Planning
        navigator = RobotNavigator(realtime_learner, neptune_client_instance)
        planner = PathPlannerRRTStar(realtime_learner, neptune_client_instance, CONFIG_SPACE_BOUNDS)
        navigator.planner = planner # Link planner to navigator

        # Federated Learning
        fl_client = FederatedLearningClient(realtime_learner.model) # Train the faster model?

        # Runtime state
        trackers = {}
        object_state_history = {} # {obj_id: deque}

        # ROS/Camera Init
        loop_logger.info("Initializing ROS2 node...")
        # Assume rclpy.init called outside
        environment_info_node = EnvironmentInfoNode()
        loop_logger.info("Opening video capture...")
        capture = cv2.VideoCapture(0) # Check index
        if not capture.isOpened(): raise IOError("Cannot open webcam")

        robot_id = f'robot_{random.randint(1000, 9999)}'
        frame_count = 0
        last_fl_check_time = time.time(); fl_check_interval = 60
        loop_logger.info("Initialization complete. Starting loop: %s", robot_id)

    except Exception as e: loop_logger.critical("LOOP INIT FAILED: %s.", e, exc_info=True); return # Exit if critical init fails

    # --- Main Loop ---
    try:
        while rclpy.ok(): # Use rclpy.ok() to allow ROS shutdown signal
            loop_start_time = time.time()
            rclpy.spin_once(environment_info_node, timeout_sec=0.01) # Process ROS callbacks

            ret, frame = capture.read()
            if not ret: loop_logger.warning("Frame read failed."); time.sleep(0.1); continue
            frame_count += 1

            try: # Inner try block for core loop logic robustness
                detections = detect_objects(frame, object_detector)
                current_robot_state = get_robot_state()
                current_environment_info_comb = get_environment_info_for_combiner()
                current_obstacles_comb = process_lidar_data_for_obstacles(environment_info_global)

                processed_object_ids=set(); realtime_object_predictions={}; object_actual_states={}; current_frame_fl_data=[]

                # --- Tracking & Prediction Update ---
                for detection in detections:
                     try: # Process each detection independently
                        object_id=detection.get('id'); measurement=detection.get('measurement')
                        if object_id is None or measurement is None: continue
                        processed_object_ids.add(object_id)
                        # Kalman
                        if object_id not in trackers: trackers[object_id] = KalmanTracker(np.array([measurement[0], measurement[1], 0.0, 0.0])); object_state_history[object_id] = deque(maxlen=HISTORY_LENGTH)
                        updated_state_kf = trackers[object_id].update(measurement)
                        if updated_state_kf is None: continue
                        current_state_flat = updated_state_kf.flatten(); object_actual_states[object_id] = current_state_flat; current_timestamp_iso = datetime.now(timezone.utc).isoformat()
                        # Update Neptune & Local State
                        neptune_client_instance.upsert_object_state(object_id, current_state_flat, current_timestamp_iso, label=detection.get('class'))
                        realtime_learner.update_local_cache(object_id, updated_state_kf, current_timestamp_iso)
                        object_state_history[object_id].append(current_state_flat)
                        if len(object_state_history[object_id])==HISTORY_LENGTH: current_frame_fl_data.append(list(object_state_history[object_id]))
                        # Real-time Prediction
                        if len(object_state_history[object_id])>=HISTORY_LENGTH:
                             hist_array = np.array(list(object_state_history[object_id])[-HISTORY_LENGTH:]).reshape(1, HISTORY_LENGTH, KALMAN_STATE_DIM)
                             pred_tensor = realtime_learner.predict(tf.constant(hist_array, dtype=tf.float32))
                             if pred_tensor is not None: realtime_object_predictions[object_id] = pred_tensor[0, -1, :].numpy()
                     except (ValueError, TypeError, Exception) as e_det: loop_logger.error("Proc Detect %s Error: %s", detection.get('id'), e_det)

                # Remove Old Trackers
                lost_ids = set(trackers.keys()) - processed_object_ids
                for lost_id in lost_ids: loop_logger.info("Obj %s lost.", lost_id); del trackers[lost_id]; del object_state_history[lost_id]

                # --- Uncertainty ---
                try: uncertainty_consultant.update_uncertainty(object_state_history)
                except Exception as e: loop_logger.error("Uncertainty update error: %s", e)
                metrics = uncertainty_consultant.get_metrics(); avg_var = metrics.get("avg_pos_var", 0.0); obj_vars = metrics.get("obj_vars", {})
                try: planner.adjust_safety_margins(avg_var, obj_vars); navigator.set_cautiousness(avg_var)
                except Exception as e: loop_logger.error("Error adjusting params: %s", e)

                # --- Combiner ---
                waypoints = None
                if realtime_object_predictions:
                    try: # Prepare inputs (NEEDS REFINEMENT)
                         avg_pred = np.mean(list(realtime_object_predictions.values()), axis=0) if realtime_object_predictions else np.zeros(KALMAN_STATE_DIM)
                         conf = np.array([0.8], dtype=np.float32) # Dummy conf
                         inputs = [tf.constant(avg_pred.reshape(1,-1)), tf.constant(conf.reshape(1,-1)), tf.constant(current_environment_info_comb.reshape(1,-1)), tf.constant(current_robot_state.reshape(1,-1)), tf.constant(current_obstacles_comb.reshape(1,-1))]
                         inputs = [tf.cast(i, tf.float32) for i in inputs] # Ensure float32
                         wp_tensor = combiner(inputs, training=False)
                         if wp_tensor is not None: waypoints = wp_tensor.numpy()[0]
                    except Exception as e: loop_logger.error("Combiner predict failed: %s", e)

                # --- Navigation ---
                user_command = get_user_command()
                if user_command:
                    try: navigator.update_context(current_robot_state, current_environment_info_comb); navigator.run_navigation(user_command, object_state_history) # Pass history
                    except Exception as e: loop_logger.error("Nav command error: %s", e)
                elif waypoints is not None:
                    try: smoothed = smooth_path(waypoints); execute_path(smoothed, cautiousness=navigator.get_cautiousness())
                    except Exception as e: loop_logger.error("Exec path error: %s", e)
                # else: loop_logger.debug("Idle.")

                # --- Federated Learning ---
                current_time = time.time()
                if TFF_AVAILABLE and current_time - last_fl_check_time > fl_check_interval:
                    loop_logger.info("Running FL check...")
                    if current_frame_fl_data:
                         try: fl_client.run_fl_round(current_frame_fl_data)
                         except Exception as e: loop_logger.error("FL round error: %s", e)
                    last_fl_check_time = current_time

            # --- Catch Iteration Error ---
            except Exception as e_iter: loop_logger.error("Unhandled error in loop iter %d: %s", frame_count, e_iter, exc_info=True); time.sleep(1)

            # --- Loop Timing/Control ---
            loop_duration = time.time() - loop_start_time
            # loop_logger.debug("Loop %d duration: %.3fs", frame_count, loop_duration)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break # Optional quit key

    # --- Loop Exit Handling ---
    except KeyboardInterrupt: loop_logger.info("Loop interrupted by user (Ctrl+C).")
    except Exception as e: loop_logger.critical("Critical unhandled error in robot_loop: %s", e, exc_info=True)
    finally: # --- Cleanup ---
        loop_logger.info("Shutting down resources...")
        if capture and capture.isOpened(): capture.release(); cv2.destroyAllWindows()
        if environment_info_node: loop_logger.info("Destroying ROS2 node..."); environment_info_node.destroy_node()
        # rclpy shutdown called in main guard
        loop_logger.info("Loop cleanup complete.")


# --- Main Execution Guard ---
if __name__ == '__main__':
    logger.info("=================================================================")
    logger.info("=== Starting Robot Application (Neptune + FL + BNN Consultant) ===")
    logger.info("=================================================================")
    neptune_client_main = None; node_initialized = False
    try:
        rclpy.init(args=None); node_initialized = True; logger.info("ROS2 Initialized.")
        neptune_client_main = NeptuneAPIClientPlaceholder(api_endpoint=NEPTUNE_API_ENDPOINT)
        robot_loop(neptune_client_instance=neptune_client_main) # Pass client instance
    except SystemExit as e: logger.warning("Application exited via sys.exit() with code %s", e.code)
    except Exception as e: logger.critical("Application crashed at top level: %s", e, exc_info=True)
    finally:
        if node_initialized and rclpy.ok(): # Check if init happened and shutdown is needed
             logger.info("Shutting down ROS2 from main guard...")
             try: rclpy.shutdown()
             except Exception as e: logger.error("Error shutting down ROS2 from main: %s", e)
        logger.info("========================================")
        logger.info("=== Robot Application Finished        ===")
        logger.info("========================================")
        logging.shutdown() # Flush/close logging handlers
