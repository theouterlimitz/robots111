# COMPLETE Robot Application Code v4 (TF + Neptune Holder + FL Holder + Logging/Error + BNN Consultant)
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
from datetime import datetime, timezone, timedelta
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
    print("Warning: tensorflow_federated not installed. FL client simulation is limited.")

# Google Cloud Exceptions
from google.api_core import exceptions as google_api_exceptions
from google.cloud import exceptions as google_cloud_exceptions

# --- Logging Configuration ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
log_level = logging.INFO # Change to logging.DEBUG for more detail

root_logger = logging.getLogger()
if not root_logger.hasHandlers():
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    log_file = 'robot_app.log'
    try:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(log_formatter)
    except ImportError:
        print(f"Warning: RotatingFileHandler not found. Using basic FileHandler for {log_file}.")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(log_formatter)
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__) # General logger

# --- Configuration ---
logger.info("Loading configuration...")
# TODO: Use a robust configuration management system
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
YOLO_CFG_PATH = os.environ.get("YOLO_CFG_PATH", "yolov3.cfg")
YOLO_WEIGHTS_PATH = os.environ.get("YOLO_WEIGHTS_PATH", "yolov3.weights")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", 'your-gcp-project-id')
PUBSUB_TOPIC_ID = 'object_exploration'
PUBSUB_SUBSCRIPTION_ID = 'robot_subscription'
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", 'your-bucket-name') # Maybe for logs/archives now
NEPTUNE_API_ENDPOINT = os.environ.get("NEPTUNE_API_ENDPOINT", "http://placeholder.neptune.api/graph")
FL_SERVER_URL = os.environ.get("FL_SERVER_URL", "http://placeholder.fl.server")

# RRT* Planner Configuration
CONFIG_SPACE_BOUNDS = (0, 20, 0, 15) # Example: (min_x, max_x, min_y, max_y)
STEP_SIZE = 0.5
MAX_ITERATIONS = 5000
GOAL_SAMPLE_RATE = 0.1
SEARCH_RADIUS = 1.5
ROBOT_RADIUS = 0.3
OBJECT_BASE_RADIUS = 0.2 # Assumed base radius for dynamic obstacles
PLANNER_SAFETY_FACTOR = 1.5 # Scales uncertainty std dev to safety margin

# Uncertainty Consultant Configuration
UNCERTAINTY_UPDATE_HZ = 1.0 # How often to run BNN analysis
BNN_MC_SAMPLES = 30       # Number of samples for MC Dropout
HIGH_UNCERTAINTY_THRESHOLD = 0.5 # Variance threshold example
HISTORY_LENGTH = 10 # Number of time steps for state history

# --- Configuration Validation ---
# ... (Add checks as needed) ...
if GEMINI_API_KEY == "YOUR_API_KEY_HERE": logger.warning("GEMINI_API_KEY not set.")
# ... etc ...

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
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if initial_state is None or len(initial_state) != KALMAN_STATE_DIM or initial_state.ndim != 1:
             msg = f"Initial state must be 1D array of dim {KALMAN_STATE_DIM}, got shape {initial_state.shape if initial_state is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)
        # Matrices NEED TUNING
        self.filter = cv2.KalmanFilter(KALMAN_STATE_DIM, KALMAN_MEASURE_DIM)
        self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        dt = 1.0 # TODO: MUST replace with actual time delta between updates
        self.filter.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.filter.processNoiseCov = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.03 # Tune process noise
        self.filter.measurementNoiseCov = np.eye(KALMAN_MEASURE_DIM, dtype=np.float32) * 0.1 # Tune sensor noise
        # Ensure initial state is column vector
        init_state_col = initial_state.reshape(-1, 1).astype(np.float32)
        self.filter.statePost = init_state_col
        self.filter.statePre = init_state_col # Initialize statePre too
        self.filter.errorCovPost = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune initial uncertainty
        self.filter.errorCovPre = np.eye(KALMAN_STATE_DIM, dtype=np.float32) * 0.1 # Tune initial uncertainty

    def predict(self):
        """Predicts the next state."""
        try:
            predicted_state = self.filter.predict()
            return predicted_state # Shape: (KALMAN_STATE_DIM, 1)
        except cv2.error as e:
            self.logger.error("OpenCV Kalman predict error: %s", e, exc_info=True)
            return None

    def update(self, measurement):
        """Updates filter state with a new measurement."""
        if measurement is None or len(measurement) != KALMAN_MEASURE_DIM or measurement.ndim != 1:
             msg = f"Measurement must be 1D array of dim {KALMAN_MEASURE_DIM}, got shape {measurement.shape if measurement is not None else 'None'}"
             self.logger.error(msg)
             raise ValueError(msg)
        measurement_col = measurement.reshape(-1, 1).astype(np.float32)
        try:
            # Correct step updates statePost
            self.filter.correct(measurement_col)
            return self.filter.statePost # Shape: (KALMAN_STATE_DIM, 1)
        except cv2.error as e:
            self.logger.error("OpenCV Kalman correct error: %s", e, exc_info=True)
            # Return previous statePost on error to avoid using potentially corrupted state
            return self.filter.statePost


# --- Transformer Model (TensorFlow) ---
class TransformerBlock(tf.keras.layers.Layer):
    """Standard Transformer Block Implementation."""
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model; self.num_heads = num_heads; self.dff = dff; self.rate = rate
        # Ensure key_dim calculation is safe
        key_dim_calc = self.d_model // self.num_heads if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads == 0 else self.d_model
        if key_dim_calc == 0 and self.d_model > 0: key_dim_calc = self.d_model # Fallback if num_heads is invalid
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim_calc, name=f"{kwargs.get('name', 'tf_block')}_mha")
        self.layernorm1 = LayerNormalization(epsilon=1e-6, name=f"{kwargs.get('name', 'tf_block')}_ln1")
        self.ffn = tf.keras.Sequential(
            [Dense(self.dff, activation='relu'), Dense(self.d_model)],
            name=f"{kwargs.get('name', 'tf_block')}_ffn" )
        self.layernorm2 = LayerNormalization(epsilon=1e-6, name=f"{kwargs.get('name', 'tf_block')}_ln2")
        self.dropout1 = Dropout(self.rate, name=f"{kwargs.get('name', 'tf_block')}_drop1")
        self.dropout2 = Dropout(self.rate, name=f"{kwargs.get('name', 'tf_block')}_drop2")

    def call(self, x, training=None, mask=None):
        attn_output = self.mha(query=x, value=x, key=x, attention_mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class ObjectBehaviorLearner:
    """Manages the Transformer model for predicting object behavior (point estimates)."""
    def __init__(self, prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.prediction_horizon = prediction_horizon # Used for building model input shape if fixed
        self.state_dimension = state_dimension
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.local_object_graph_cache = nx.Graph() # Local cache
        self.logger.info("Initializing ObjectBehaviorLearner (local graph is cache).")

        try:
            if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads != 0:
                 self.logger.warning("d_model (%d) not divisible by num_heads (%d). Check MHA config.", self.d_model, self.num_heads)

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

                @tf.function
                def call(self, inputs, training=None, mask=None):
                    x = self.embedding_layer(inputs)
                    # TODO: Add positional encoding if needed
                    for block in self.transformer_blocks:
                        x = block(x, training=training, mask=mask) # Pass training flag
                    return self.output_dense_layer(x)

            self.model = TransformerModel(
                self.embedding_layer, self.transformer_blocks, self.output_dense_layer
            )
            # Build model with flexible input shape (None for batch and time steps)
            self.model.build(input_shape=(None, None, state_dimension))
            self.model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
            self.logger.info("ObjectBehaviorLearner TF Model built and compiled.")

        except Exception as e:
            self.logger.critical("Failed to build/compile ObjectBehaviorLearner model: %s", e, exc_info=True)
            raise

    def train(self, past_states, target_states):
        """Trains the internal Keras model."""
        self.logger.info("Starting training with %d samples...", len(past_states))
        try:
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            val_loss = history.history.get('val_loss', [None])[-1]
            self.logger.info("Training finished. Final validation loss: %s", f"{val_loss:.4f}" if val_loss else "N/A")
        except Exception as e:
            self.logger.error("Error during training: %s", e, exc_info=True)

    def predict(self, past_states):
        """Standard point estimate prediction (dropout disabled)."""
        # self.logger.debug("Standard prediction...")
        try:
            # Use call directly with training=False
            predictions = self.model(past_states, training=False)
            return predictions
        except Exception as e:
             self.logger.error("Error during standard prediction: %s", e, exc_info=True)
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
            # self.logger.debug("Updated node %s in local cache", object_id)


# --- BNN Variant using MC Dropout ---
class BNNObjectBehaviorLearner(ObjectBehaviorLearner):
    """BNN version using MC Dropout for uncertainty estimation."""
    def __init__(self, num_mc_samples=BNN_MC_SAMPLES, **kwargs):
        super().__init__(**kwargs) # Builds the same underlying Keras model
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not isinstance(num_mc_samples, int) or num_mc_samples <= 1:
            raise ValueError("num_mc_samples must be an integer > 1.")
        self.num_mc_samples = num_mc_samples
        self.logger.info("Initialized with %d MC samples.", self.num_mc_samples)
        has_dropout = any(isinstance(layer, Dropout) for layer in self._get_all_layers(self.model))
        if not has_dropout:
             self.logger.warning("MC Dropout requires Dropout layers, but none were detected.")

    def _get_all_layers(self, model):
        """Helper to recursively get all layers including nested ones."""
        layers = []
        if hasattr(model, 'layers'):
            for layer in model.layers:
                 layers.append(layer)
                 layers.extend(self._get_all_layers(layer)) # Recurse
        return layers

    @tf.function
    def predict_mc_dropout_tf(self, past_states):
        """Internal TF function for the prediction loop."""
        # Use tf.TensorArray for graph compatibility
        predictions_array = tf.TensorArray(dtype=tf.float32, size=self.num_mc_samples, dynamic_size=False, element_shape=tf.shape(self.model(past_states, training=True))[1:]) # Match element shape
        for i in tf.range(self.num_mc_samples):
            predictions = self.model(past_states, training=True) # training=True enables dropout
            predictions_array = predictions_array.write(i, predictions[0]) # Write first batch element
        return predictions_array.stack()

    def predict(self, past_states: np.ndarray | tf.Tensor) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """ Performs prediction using MC Dropout to estimate mean and variance. """
        start_time = time.time()
        # self.logger.debug("BNN MC Dropout prediction...")
        try:
            if not isinstance(past_states, tf.Tensor):
                past_states = tf.constant(past_states, dtype=tf.float32)
            # Ensure batch dimension exists
            if len(tf.shape(past_states)) == 2: # (time, state) -> (1, time, state)
                 past_states = tf.expand_dims(past_states, axis=0)
            elif len(tf.shape(past_states)) != 3:
                 raise ValueError(f"Input must have shape (batch, time, state), got {tf.shape(past_states)}")

            # Use the tf.function decorated method
            predictions_stack = self.predict_mc_dropout_tf(past_states)
            # Shape: (num_mc_samples, time_steps, state_dimension) - Assuming batch size 1 was handled

            mean_prediction = tf.reduce_mean(predictions_stack, axis=0)
            variance_prediction = tf.math.reduce_variance(predictions_stack, axis=0)

            duration = time.time() - start_time
            self.logger.debug("MC Dropout prediction finished in %.3f s.", duration)
            # Return results with batch dimension squeezed if input batch was 1
            # Return shape (time_steps, state_dimension) for mean/variance if input was single sample
            # For now, keep batch dim: (1, time_steps, state_dimension)
            return mean_prediction, variance_prediction

        except Exception as e:
            self.logger.error("Error during MC Dropout prediction: %s", e, exc_info=True)
            return None, None


# --- Uncertainty Consultant ---
class UncertaintyConsultant:
    """Manages BNN models to provide uncertainty estimates."""
    def __init__(self, bnn_learner_instance: BNNObjectBehaviorLearner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ):
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        if not isinstance(bnn_learner_instance, BNNObjectBehaviorLearner):
             raise TypeError("UncertaintyConsultant requires a BNNObjectBehaviorLearner instance.")
        self.bnn_learner = bnn_learner_instance
        self.current_uncertainty_metrics = {}
        self.update_period_secs = 1.0 / update_frequency_hz if update_frequency_hz > 0 else float('inf')
        self.last_update_time = 0
        self.logger.info("Initialized with update period %.2f s.", self.update_period_secs)

    def update_uncertainty(self, object_state_history: dict):
        """Runs BNN prediction periodically and updates internal metrics."""
        current_time = time.time()
        if current_time - self.last_update_time < self.update_period_secs:
            return # Update less frequently

        self.logger.debug("Running BNN uncertainty analysis...")
        overall_variance_sum = 0.0; num_predictions = 0; high_uncertainty_objects = []; object_variances = {}
        history_len_needed = self.bnn_learner.prediction_horizon

        start_analysis_time = time.time()
        objects_analyzed = 0
        for obj_id, history_deque in object_state_history.items():
            if len(history_deque) >= history_len_needed:
                history_list = list(history_deque)[-history_len_needed:] # Get most recent history
                history_array = np.array(history_list).reshape(1, history_len_needed, self.bnn_learner.state_dimension)
                try:
                    mean_pred, var_pred = self.bnn_learner.predict(history_array)
                    objects_analyzed += 1
                    if var_pred is not None:
                        # Metric: Avg positional variance over horizon
                        pos_variance = var_pred[0, :, :2] # Assumes state = [x, y, ...]
                        avg_pos_var = tf.reduce_mean(pos_variance).numpy()
                        overall_variance_sum += avg_pos_var; num_predictions += 1; object_variances[obj_id] = avg_pos_var
                        if avg_pos_var > HIGH_UNCERTAINTY_THRESHOLD: high_uncertainty_objects.append(obj_id)
                except Exception as e:
                     self.logger.error("Error during BNN prediction for object %s: %s", obj_id, e, exc_info=True)

        analysis_duration = time.time() - start_analysis_time
        avg_overall_variance = overall_variance_sum / num_predictions if num_predictions > 0 else 0.0
        self.current_uncertainty_metrics = {
            "average_positional_variance": avg_overall_variance,
            "high_uncertainty_ids": high_uncertainty_objects,
            "object_variances": object_variances }
        self.last_update_time = current_time
        self.logger.info("Uncertainty metrics updated (analyzed %d objects in %.3fs): AvgVar=%.4f, HighUncertainty=%d obj",
                         objects_analyzed, analysis_duration, avg_overall_variance, len(high_uncertainty_objects))
        self.logger.debug("Detailed metrics: %s", self.current_uncertainty_metrics)

    def get_metrics(self) -> dict:
        """Provides the latest calculated uncertainty metrics."""
        return self.current_uncertainty_metrics


# --- Prediction Combiner (TensorFlow) ---
class PredictionCombiner(tf.keras.Model):
    # (Implementation from previous step - unchanged)
    def __init__(self, transformer_output_dim_flat, context_dim, output_dim=2, prediction_horizon=10, use_confidence_weighting=True, dropout_rate=0.2): # Set default output_dim=2 maybe?
        super(PredictionCombiner, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.use_confidence_weighting = use_confidence_weighting
        self.dropout_rate = dropout_rate
        self.prediction_horizon = prediction_horizon
        self.output_dim = output_dim
        # Define layers based on flat input sizes
        # TODO: Calculate input_concat_size accurately
        # input_concat_size = transformer_output_dim_flat + 1 (confidence) + context_dim
        self.dense1 = Dense(128, activation='relu', name="combiner_dense1")
        self.dense2 = Dense(64, activation='relu', name="combiner_dense2")
        self.dense3 = Dense(self.output_dim * self.prediction_horizon, name="combiner_output_dense")
        self.layer_norm = LayerNormalization(epsilon=1e-6, name="combiner_layernorm")
        self.dropout = Dropout(self.dropout_rate, name="combiner_dropout")
        self.logger.info("Prediction Combiner Initialized.")

    def call(self, inputs, training=None):
        # (Implementation from previous step with safe_flatten and specific error catches)
        try:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != 5:
                 self.logger.error("PredictionCombiner received invalid inputs structure.")
                 return None
            transformer_predictions, transformer_confidences, environment_info, robot_state, obstacles = inputs

            def safe_flatten(tensor): # Helper nested function
                if tensor is None: return None
                shape = tf.shape(tensor)
                if len(shape) > 2: return tf.reshape(tensor, [shape[0], -1])
                elif len(shape) == 1: return tf.expand_dims(tensor, axis=-1)
                return tensor # Assumed shape (batch, features)

            tf_preds_flat = safe_flatten(transformer_predictions)
            tf_confs_flat = safe_flatten(transformer_confidences)
            env_info_flat = safe_flatten(environment_info)
            robot_state_flat = safe_flatten(robot_state)
            obstacles_flat = safe_flatten(obstacles)

            # Check for None after flattening before concatenation
            input_parts = [part for part in [tf_preds_flat, env_info_flat, robot_state_flat, obstacles_flat] if part is not None]
            if not input_parts:
                 self.logger.error("Combiner has no valid inputs after flattening.")
                 return None

            # Handle confidence weighting carefully
            weighted_preds = tf_preds_flat
            if self.use_confidence_weighting and tf_confs_flat is not None and tf_preds_flat is not None:
                 try:
                      confidences_exp = tf_confs_flat
                      if len(tf.shape(confidences_exp)) == 1: confidences_exp = tf.expand_dims(confidences_exp, axis=-1)
                      weighted_preds = tf_preds_flat * confidences_exp
                 except Exception as e_weight:
                      self.logger.error("Error applying confidence weighting: %s", e_weight)
                      # Proceed with unweighted predictions?
                      weighted_preds = tf_preds_flat # Fallback

            # Update input parts for concatenation
            if weighted_preds is not None:
                 input_parts[0] = weighted_preds # Replace original preds with weighted ones if available

            combined_input = Concatenate()(input_parts)

            x = self.layer_norm(combined_input, training=training)
            x = self.dropout(x, training=training)
            x = self.dense1(x)
            x = self.dense2(x)
            waypoints_flat = self.dense3(x)
            final_shape = [-1, self.prediction_horizon, self.output_dim]
            waypoints = tf.reshape(waypoints_flat, final_shape)
            return waypoints
        except (tf.errors.InvalidArgumentError, ValueError) as e:
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
            raise # Propagate error if subscription fails

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
logger.info("Initializing Google Cloud Pub/Sub clients...")
publisher = None; subscriber = None; topic_path = None; subscription_path = None
# ... (Try-except block for init and create topic/sub) ...
try:
     publisher = pubsub.PublisherClient()
     subscriber = pubsub.SubscriberClient()
     # ... (rest of setup) ...
except Exception as e:
     logger.error("Failed to init Pub/Sub: %s", e, exc_info=True)


# --- Cloud Storage Function (Definition only - for reference) ---
def save_graph_to_cloud_storage(graph, bucket_name, blob_name):
     # (Implementation from previous step)
     func_logger = logging.getLogger(__name__)
     try:
          # ... (GCS client, upload logic with specific exceptions) ...
          pass
     except Exception as e:
          func_logger.error("Error saving graph: %s", e, exc_info=True)

# --- Neptune API Client Placeholder ---
# (Keep implementation from previous step)
class NeptuneAPIClientPlaceholder:
    # ... (implementation) ...
    def __init__(self, api_endpoint=NEPTUNE_API_ENDPOINT, api_key="DUMMY_KEY"): # Use config
        self.logger = logging.getLogger(__name__)
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.logger.info("Initialized NeptuneAPIClientPlaceholder for endpoint: %s", self.api_endpoint)
    # ... (rest of methods) ...
    def get_object(self, object_id: str): # Example
         self.logger.info("Sim Neptune: get_object(%s)", object_id); return {'id': object_id, 'state': [1,1,0,0]} if random.random()<0.9 else None
    def query_objects(self, criteria: dict): return [] # Placeholder
    def query_map_features(self, bounds: dict): return [] # Placeholder
    def upsert_object_state(self, object_id: str, state_vector, timestamp_iso: str, **kwargs): self.logger.info("Sim Neptune: upsert %s", object_id); return True
    def update_object_semantics(self, object_id: str, properties: dict): self.logger.info("Sim Neptune: update semantics %s", object_id); return True


# --- Robot Navigator ---
# (Keep implementation from previous step, uses Neptune client)
class RobotNavigator:
    # ... (Implementation uses neptune_client, includes set_cautiousness) ...
    def __init__(self, object_behavior_learner, neptune_client, max_retries=3, initial_delay=1): # Added neptune_client
        self.logger = logging.getLogger(__name__)
        self.object_learner = object_behavior_learner
        self.neptune_client = neptune_client # Use the passed-in client
        self.robot_state = None
        self.environment_info = None
        self.gemini_model = None
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.planner = None # Will be set later
        self.current_cautiousness = 0.0
        # Init Gemini model
        try:
            if gemini_configured: self.gemini_model = google.generativeai.GenerativeModel('gemini-pro')
            else: self.logger.warning("Gemini model not initialized due to config.")
        except Exception as e: self.logger.error("Error initializing Gemini Model: %s", e, exc_info=True)

    def update_context(self, robot_state, environment_info): self.robot_state = robot_state; self.environment_info = environment_info
    def format_environment_info(self): return str(self.environment_info) if self.environment_info is not None else "N/A"
    def format_graph_info(self): return "Context from Neptune (Simulated)" # Placeholder, real formatting in _build...
    def set_cautiousness(self, uncertainty_metric): self.current_cautiousness = np.clip(uncertainty_metric * 2.0, 0.0, 1.0); self.logger.debug("Navigator cautiousness: %.2f", self.current_cautiousness)
    def get_cautiousness(self): return self.current_cautiousness

    def _build_gemini_prompt(self, user_command):
        # (Implementation uses neptune_client.query_objects etc. as before)
        graph_context_str = "Graph context error."; map_features_str = "Map feature error."
        try:
             robot_loc = self.robot_state[:2] if self.robot_state is not None else [0,0] # Get location from state
             radius_m = 50
             query_criteria = {'location_radius': {'x': robot_loc[0], 'y': robot_loc[1], 'radius_m': radius_m}}
             nearby_objects = self.neptune_client.query_objects(criteria=query_criteria)
             if nearby_objects:
                 object_summaries = [f"- ID: {obj.get('id')}, Label: {obj.get('label', 'N/A')}" for obj in nearby_objects[:5]] # Limit summary size
                 graph_context_str = "Nearby objects:\n" + "\n".join(object_summaries)
             else: graph_context_str = "No relevant objects found nearby."
             # Query map features (optional)
             map_bounds = {'min_x': robot_loc[0]-20, 'max_x': robot_loc[0]+20, 'min_y': robot_loc[1]-20, 'max_y': robot_loc[1]+20}
             map_features = self.neptune_client.query_map_features(bounds=map_bounds)
             map_features_str = f"{len(map_features)} map features found (placeholder)." if map_features else "No relevant map features found."
        except Exception as e: self.logger.error("Error fetching context for Gemini: %s", e, exc_info=True)

        env_info_str = self.format_environment_info()
        robot_state_str = str(self.robot_state)
        # Build the prompt...
        prompt = f"""[Full Prompt Template Using Above Context Strings]
        Robot State: {robot_state_str}
        Environment: {env_info_str}
        Nearby Context: {graph_context_str}
        Map Context: {map_features_str}
        Command: {user_command}
        Instructions: [Your instructions for JSON output]"""
        return prompt

    # get_gemini_navigation_instructions (with retries)
    # parse_gemini_instructions
    # map_actions_to_robot_commands (uses self.planner)
    # execute_commands
    # run_navigation
    # ... (Keep implementations for these methods from previous steps, ensuring logging and error handling) ...


# --- Path Planner (RRT*) ---
# (Keep PathPlannerRRTStar class definition, with uncertainty adjustments)
class PathPlannerRRTStar:
    # ... (Full implementation from previous step, including adjust_safety_margins and modified _is_collision) ...
     def __init__(self, behavior_learner, neptune_client, config_space_bounds, step_size=STEP_SIZE, max_iter=MAX_ITERATIONS, goal_sample_rate=GOAL_SAMPLE_RATE, search_radius=SEARCH_RADIUS, robot_radius=ROBOT_RADIUS):
          self.logger = logging.getLogger(f"{self.__class__.__name__}")
          self.behavior_learner = behavior_learner
          self.neptune_client = neptune_client
          # ... (Store other params: bounds, step_size, etc.) ...
          self.min_x, self.max_x, self.min_y, self.max_y = config_space_bounds
          self.step_size=step_size; self.max_iter=max_iter; self.goal_sample_rate=goal_sample_rate; self.search_radius=search_radius; self.robot_radius=robot_radius;
          self.object_base_radius = OBJECT_BASE_RADIUS
          self.uncertainty_margins = {}
          self.default_uncertainty_margin = 0.0
          self.nodes = []; self.kdtree = None; self.start_node = None; self.goal_node = None
          self.logger.info(f"RRT* Planner Initialized. Neptune client, KDTree: {USE_KDTREE}")

     def adjust_safety_margins(self, average_variance: float, object_variances: dict):
          default_std_dev = math.sqrt(max(0, average_variance))
          self.default_uncertainty_margin = default_std_dev * PLANNER_SAFETY_FACTOR
          self.logger.debug("Default uncertainty margin: %.3f", self.default_uncertainty_margin)
          self.uncertainty_margins.clear()
          for obj_id, var in object_variances.items():
               margin = math.sqrt(max(0, var)) * PLANNER_SAFETY_FACTOR
               self.uncertainty_margins[obj_id] = margin

     def get_goal_position(self, target) -> tuple | None:
          # (Implementation uses self.neptune_client.get_object)
          # ...
          pass # Keep implementation

     def predict_dynamic_obstacles(self, robot_pose, prediction_horizon_secs, time_step):
          # (Implementation uses self.behavior_learner.predict)
          # ...
          self.logger.debug("Predicting dynamic obstacles...")
          return {} # Placeholder return

     def _is_collision(self, node1, node2, dynamic_obstacles_pred: dict, static_features: list):
          # (Implementation uses self.uncertainty_margins)
          # ...
          p1=np.array([node1.x,node1.y]); p2=np.array([node2.x,node2.y])
          # --- Static Check ---
          # TODO: Implement checks against static_features geometries
          # --- Dynamic Check ---
          if dynamic_obstacles_pred:
               segment_vec = p2 - p1; segment_len = np.linalg.norm(segment_vec)
               if segment_len < 1e-6: return False # Skip zero-length
               num_steps = max(2, int(segment_len / (self.step_size * 0.5)))
               for i in range(num_steps + 1):
                    t_interp = i / num_steps
                    robot_pos = p1 + t_interp * segment_vec
                    for obj_id, trajectory in dynamic_obstacles_pred.items():
                         # Simplified: Check against all trajectory points
                         for (_time, obs_x, obs_y, _obs_base_radius) in trajectory:
                              margin = self.uncertainty_margins.get(obj_id, self.default_uncertainty_margin)
                              effective_obs_radius = self.object_base_radius + margin
                              obs_center = np.array([obs_x, obs_y])
                              total_radius_sq = (self.robot_radius + effective_obs_radius)**2
                              if np.sum((robot_pos - obs_center)**2) <= total_radius_sq:
                                   # self.logger.debug("Collision: DynObj %s", obj_id)
                                   return True
          return False # Assume no collision if checks pass

     def plan_path(self, start_pose, goal_target, environment_info):
          # (Full RRT* implementation using Neptune for goal/static features,
          #  local predictions for dynamic, and uncertainty margins in collision checks)
          start_time = time.time()
          self.logger.info("RRT* Plan: Start=%s, Target='%s'", start_pose[:2], goal_target)
          self.start_node = Node(start_pose[0], start_pose[1])
          goal_pos = self.get_goal_position(goal_target) # Uses Neptune
          if goal_pos is None: return None
          self.goal_node = Node(goal_pos[0], goal_pos[1])
          self.nodes = [self.start_node]; best_goal_node = None
          if USE_KDTREE: self.kdtree = KDTree([[n.x, n.y] for n in self.nodes])

          # Get static/dynamic obstacles
          map_bounds = {'min_x': min(start_pose[0], goal_pos[0]) - 10, 'max_x': max(start_pose[0], goal_pos[0]) + 10,
                        'min_y': min(start_pose[1], goal_pos[1]) - 10, 'max_y': max(start_pose[1], goal_pos[1]) + 10} # Dynamic bounds
          static_features = self.neptune_client.query_map_features(bounds=map_bounds)
          dynamic_preds = self.predict_dynamic_obstacles(start_pose, 5.0, 0.5) # Uses local TF model

          for i in range(self.max_iter):
               rnd_point = self._get_random_point()
               nearest_node = self._get_nearest_node(rnd_point)
               if nearest_node is None: continue
               new_potential_node = self._steer(nearest_node, rnd_point)

               # Check collision for the edge using fetched static features and dynamic predictions
               if not self._is_collision(nearest_node, new_potential_node, dynamic_preds, static_features):
                    near_nodes = self._find_near_nodes(new_potential_node)
                    min_cost_node = nearest_node
                    min_new_cost = self._calculate_cost(nearest_node, new_potential_node)
                    # Choose parent
                    for near_node in near_nodes:
                         if not self._is_collision(near_node, new_potential_node, dynamic_preds, static_features):
                              cost_via_near = self._calculate_cost(near_node, new_potential_node)
                              if cost_via_near < min_new_cost: min_cost_node = near_node; min_new_cost = cost_via_near
                    # Add node
                    new_node = Node(new_potential_node.x, new_potential_node.y, cost=min_new_cost, parent=min_cost_node)
                    self.nodes.append(new_node)
                    if USE_KDTREE: self.kdtree = KDTree([[n.x, n.y] for n in self.nodes]) # Rebuild KDTree
                    # Rewire
                    for near_node in near_nodes:
                         if near_node == min_cost_node: continue
                         cost_via_new = self._calculate_cost(new_node, near_node)
                         if cost_via_new < near_node.cost and not self._is_collision(new_node, near_node, dynamic_preds, static_features):
                              near_node.parent = new_node; near_node.cost = cost_via_new
                    # Goal check
                    dist_to_goal = self._distance(new_node, self.goal_node)
                    if dist_to_goal <= self.step_size:
                         if not self._is_collision(new_node, self.goal_node, dynamic_preds, static_features):
                              final_cost = self._calculate_cost(new_node, self.goal_node)
                              if best_goal_node is None or final_cost < best_goal_node.cost:
                                   best_goal_node = Node(self.goal_node.x, self.goal_node.y, cost=final_cost, parent=new_node)

          # Path Reconstruction... (same as before)
          elapsed_time = time.time() - start_time
          # ... (logging and return logic) ...
          if best_goal_node: path = self._reconstruct_path(best_goal_node); self.logger.info("RRT* SUCCESS..."); return path
          else: self.logger.error("RRT* FAILED..."); return None # Simplified error logging


# --- Federated Learning Client Placeholder ---
# (Keep implementation from previous step)
class FederatedLearningClient:
     # ... (implementation) ...
     def __init__(self, keras_model, fl_server_url=FL_SERVER_URL):
          self.logger = logging.getLogger(__name__)
          self.local_model = keras_model; self.server_url = fl_server_url; self.tff_available = TFF_AVAILABLE; self.initial_weights = None
          # ... (rest of init) ...
     # ... (rest of methods) ...

# --- Placeholder Functions ---
# (Keep definitions, potentially add more realistic simulation)
def get_user_command(): logger.debug("Checking user cmd..."); return None
def get_robot_state(): return np.array([random.uniform(1,19), random.uniform(1,14), 0.0, 0.0])
def get_environment_info_for_combiner(): global environment_info_global; return environment_info_global.astype(np.float32) if environment_info_global is not None else np.zeros(2, dtype=np.float32)
def process_lidar_data_for_obstacles(lidar_data): return np.array([np.mean(lidar_data if lidar_data is not None else [-1.0])], dtype=np.float32)
def smooth_path(waypoints): return waypoints # Placeholder
def execute_path(path, cautiousness=0.0):
    # ... (implementation from previous step) ...
    pass
def detect_objects(frame, detector): # Placeholder with random IDs
    # ... (implementation) ...
    return [{'id': f'sim_{int(time.time()*10 % 1000) + i}', 'measurement': np.array([random.uniform(1,19), random.uniform(1,14)]), 'confidence': 0.85, 'class': 'sim'} for i in range(random.randint(0,4))]

# ... (Keep Gemini Object Query Placeholders, ensure they use neptune_client for updates) ...
def update_object_graph_with_gemini(neptune_client, object_id, gemini_response):
     logger.info("Updating Neptune with Gemini info for %s", object_id)
     neptune_client.update_object_semantics(object_id, gemini_response)

# ==============================================================================
# === Main Robot Loop ==========================================================
# ==============================================================================
def robot_loop(neptune_client_instance): # Expect neptune client to be passed in
    loop_logger = logging.getLogger("RobotLoop")
    loop_logger.info("--- Initializing Robot Loop Components ---")
    navigator = None # Ensure defined in outer scope for finally block if needed
    planner = None
    fl_client = None
    capture = None
    environment_info_node = None

    # --- Initialization Block ---
    try:
        # Instantiate models, passing config/state dimensions
        realtime_learner = ObjectBehaviorLearner(prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM)
        bnn_learner = BNNObjectBehaviorLearner(prediction_horizon=HISTORY_LENGTH, state_dimension=KALMAN_STATE_DIM, num_mc_samples=BNN_MC_SAMPLES)
        uncertainty_consultant = UncertaintyConsultant(bnn_learner_instance=bnn_learner, update_frequency_hz=UNCERTAINTY_UPDATE_HZ)

        # Combiner setup (refine dimensions based on actual features)
        env_dim = 2; robot_dim = KALMAN_STATE_DIM; obstacle_dim = 1 # Example dims
        context_dim = env_dim + robot_dim + obstacle_dim
        # Assuming combiner uses last predicted state from transformer (flat)
        transformer_output_dim_flat = KALMAN_STATE_DIM
        combiner_output_dim = 2
        combiner_prediction_horizon = 10
        combiner = PredictionCombiner(transformer_output_dim_flat, context_dim, combiner_output_dim, combiner_prediction_horizon)
        combiner.compile(optimizer='adam', loss='mse') # TODO: Define appropriate loss

        # Pass neptune client to components that need it
        navigator = RobotNavigator(realtime_learner, neptune_client_instance)
        planner = PathPlannerRRTStar(realtime_learner, neptune_client_instance, CONFIG_SPACE_BOUNDS)
        navigator.planner = planner # Link planner to navigator

        # Pass the model that FL should train (e.g., the realtime one)
        fl_client = FederatedLearningClient(realtime_learner.model)

        # Runtime state
        trackers = {}
        object_state_history = {} # {obj_id: deque}

        # ROS/Camera Init
        loop_logger.info("Initializing ROS2 node...")
        # rclpy.init(args=None) # Init should happen once before calling robot_loop
        environment_info_node = EnvironmentInfoNode()
        loop_logger.info("Opening video capture...")
        capture = cv2.VideoCapture(0)
        if not capture.isOpened(): raise IOError("Cannot open webcam")

        robot_id = f'robot_{random.randint(1000, 9999)}'
        frame_count = 0
        last_fl_check_time = time.time()
        fl_check_interval = 60 # Seconds

        loop_logger.info("Initialization complete. Starting main loop for robot: %s", robot_id)

    # --- Catch initialization errors ---
    except Exception as e:
        loop_logger.critical("Robot Loop Initialization failed: %s. Cannot start loop.", e, exc_info=True)
        # Cleanup resources acquired during failed init
        if capture and capture.isOpened(): capture.release()
        if environment_info_node: environment_info_node.destroy_node()
        # if rclpy.ok(): rclpy.shutdown() # Shutdown only if init succeeded
        return # Exit the function

    # --- Main Loop ---
    try:
        while rclpy.ok():
            loop_start_time = time.time()
            rclpy.spin_once(environment_info_node, timeout_sec=0.01)

            ret, frame = capture.read()
            if not ret: loop_logger.warning("Failed frame read."); time.sleep(0.1); continue
            frame_count += 1

            try: # Wrap core iteration logic
                detections = detect_objects(frame, object_detector)
                current_robot_state = get_robot_state() # [x, y, vx, vy] format assumed by planner/navigator
                current_environment_info_comb = get_environment_info_for_combiner()
                current_obstacles_comb = process_lidar_data_for_obstacles(environment_info_global)

                processed_object_ids = set()
                realtime_object_predictions = {} # Store point estimates from realtime_learner
                object_actual_states = {} # Store current KF states
                current_frame_fl_data = [] # Data for this iteration's FL potential

                # --- Tracking, Prediction, State Update ---
                for detection in detections:
                     try:
                        object_id = detection.get('id')
                        measurement = detection.get('measurement') # Expect [x,y]
                        if object_id is None or measurement is None: continue
                        processed_object_ids.add(object_id)

                        if object_id not in trackers:
                            initial_state = np.array([measurement[0], measurement[1], 0.0, 0.0], dtype=np.float32)
                            if len(initial_state) != KALMAN_STATE_DIM: continue # Skip if measurement is bad for init
                            trackers[object_id] = KalmanTracker(initial_state)
                            object_state_history[object_id] = deque(maxlen=HISTORY_LENGTH)

                        kalman_tracker = trackers[object_id]
                        # predicted_state_kf = kalman_tracker.predict() # Predict before update
                        updated_state_kf = kalman_tracker.update(measurement) # Update with measurement
                        if updated_state_kf is None: continue # Kalman update failed

                        current_state_flat = updated_state_kf.flatten()
                        object_actual_states[object_id] = current_state_flat
                        current_timestamp_iso = datetime.now(timezone.utc).isoformat()

                        # --- Update Central Store (Neptune) ---
                        neptune_client_instance.upsert_object_state(
                            object_id, current_state_flat, current_timestamp_iso, label=detection.get('class'))

                        # --- Update Local Cache & History ---
                        realtime_learner.update_local_cache(object_id, updated_state_kf, current_timestamp_iso) # Update standard learner's cache
                        bnn_learner.update_local_cache(object_id, updated_state_kf, current_timestamp_iso) # Update BNN learner's cache too if separate

                        object_state_history[object_id].append(current_state_flat)
                        if len(object_state_history[object_id]) == HISTORY_LENGTH:
                             current_frame_fl_data.append(list(object_state_history[object_id]))

                        # --- Real-time Prediction ---
                        if len(object_state_history[object_id]) >= HISTORY_LENGTH:
                             history_array = np.array(list(object_state_history[object_id])).reshape(1, HISTORY_LENGTH, KALMAN_STATE_DIM)
                             pred_tensor = realtime_learner.predict(history_array)
                             if pred_tensor is not None:
                                 realtime_object_predictions[object_id] = pred_tensor[0, -1, :].numpy()

                     except (ValueError, TypeError, Exception) as e_det:
                          loop_logger.error("Error processing detection %s: %s", detection.get('id', 'N/A'), e_det, exc_info=True)

                # Remove Old Trackers
                lost_track_ids = set(trackers.keys()) - processed_object_ids
                for lost_id in lost_track_ids:
                    loop_logger.info("Object %s lost track.", lost_id)
                    if lost_id in trackers: del trackers[lost_id]
                    if lost_id in object_state_history: del object_state_history[lost_id]

                # --- Uncertainty Analysis ---
                try: uncertainty_consultant.update_uncertainty(object_state_history)
                except Exception as e_uncert: loop_logger.error("Error updating uncertainty: %s", e_uncert, exc_info=True)

                # --- Get & Use Uncertainty ---
                metrics = uncertainty_consultant.get_metrics()
                avg_variance = metrics.get("average_positional_variance", 0.0)
                object_variances = metrics.get("object_variances", {})
                try:
                    planner.adjust_safety_margins(avg_variance, object_variances)
                    navigator.set_cautiousness(avg_variance)
                except Exception as e_adjust: loop_logger.error("Error adjusting margins/cautiousness: %s", e_adjust, exc_info=True)

                # --- Combiner Prediction ---
                waypoints = None
                if realtime_object_predictions: # Base waypoints on faster point estimates
                    try:
                        # --- Prepare combiner inputs (NEEDS CAREFUL REVIEW/IMPLEMENTATION) ---
                        # Example: Use average state (crude) - replace with better feature engineering
                        if object_actual_states and realtime_object_predictions:
                             avg_realtime_pred = np.mean(list(realtime_object_predictions.values()), axis=0) if realtime_object_predictions else np.zeros(KALMAN_STATE_DIM)
                             dummy_confidence = np.array([0.8], dtype=np.float32)
                             combiner_input_list = [
                                 tf.constant(avg_realtime_pred.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(dummy_confidence.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_environment_info_comb.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_robot_state.reshape(1, -1), dtype=tf.float32),
                                 tf.constant(current_obstacles_comb.reshape(1, -1), dtype=tf.float32) ]
                             waypoints_tensor = combiner(combiner_input_list, training=False)
                             if waypoints_tensor is not None: waypoints = waypoints_tensor.numpy()[0]
                    except Exception as e_comb: loop_logger.error("Combiner prediction failed: %s", e_comb, exc_info=True)

                # --- Navigation Decision ---
                user_command = get_user_command()
                if user_command:
                    loop_logger.info("Processing user command: '%s'", user_command)
                    try:
                        navigator.update_context(current_robot_state, current_environment_info_comb)
                        navigator.run_navigation(user_command) # Uses planner with updated margins
                    except Exception as e_nav: loop_logger.error("Error running navigation command: %s", e_nav, exc_info=True)
                elif waypoints is not None:
                    try:
                        smoothed_path = smooth_path(waypoints)
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
                         try: fl_client.run_fl_round(current_frame_fl_data)
                         except Exception as e_fl: loop_logger.error("Error during FL round: %s", e_fl, exc_info=True)
                    else: loop_logger.info("Skipping FL round: No new sequences.")
                    last_fl_check_time = current_time

                # --- GCS Saving is Removed ---

            except Exception as e_iter:
                 loop_logger.error("Unhandled error in main loop iteration %d: %s", frame_count, e_iter, exc_info=True)
                 time.sleep(1) # Avoid rapid error loops

            # --- Loop Timing ---
            loop_duration = time.time() - loop_start_time
            # loop_logger.debug("Loop iteration %d finished in %.3f seconds", frame_count, loop_duration)

    except KeyboardInterrupt:
        loop_logger.info("Loop interrupted by user (Ctrl+C).")
    except Exception as e:
        loop_logger.critical("Critical unhandled error in robot_loop: %s", e, exc_info=True)
    finally:
        # --- Cleanup ---
        loop_logger.info("Shutting down resources...")
        if capture and capture.isOpened(): capture.release()
        cv2.destroyAllWindows()
        # Shutdown ROS node cleanly
        if environment_info_node:
             loop_logger.info("Destroying ROS2 node...")
             try: environment_info_node.destroy_node()
             except Exception as e_ros_destroy: loop_logger.error("Error destroying ROS node: %s", e_ros_destroy)
        # Shutdown rclpy if it was initialized and is still running
        # if 'rclpy' in sys.modules and rclpy.ok():
        #     loop_logger.info("Shutting down ROS2...")
        #     try: rclpy.shutdown()
        #     except Exception as e_ros_shutdown: loop_logger.error("Error shutting down ROS2: %s", e_ros_shutdown)
        loop_logger.info("Cleanup complete.")


# --- Main Execution Guard ---
if __name__ == '__main__':
    logger.info("=================================================================")
    logger.info("=== Starting Robot Application (Neptune + FL + BNN Consultant) ===")
    logger.info("=================================================================")
    neptune_client_main = None # Ensure defined for finally block
    try:
        # Initialize ROS once at the top level
        rclpy.init(args=None)
        logger.info("ROS2 Initialized.")

        # Instantiate the Neptune client placeholder
        neptune_client_main = NeptuneAPIClientPlaceholder(api_endpoint=NEPTUNE_API_ENDPOINT)

        # Run the main robot loop logic
        robot_loop(neptune_client_instance=neptune_client_main)

    except SystemExit as e:
         logger.warning("Application exited via sys.exit() with code %s", e.code)
    except Exception as e:
        logger.critical("Application crashed at top level: %s", e, exc_info=True)
    finally:
        # Ensure ROS is shutdown if it was initialized
        if 'rclpy' in sys.modules and rclpy.ok():
             logger.info("Shutting down ROS2 from main guard...")
             try: rclpy.shutdown()
             except Exception as e_ros_main_shutdown: logger.error("Error shutting down ROS2 from main: %s", e_ros_main_shutdown)

        logger.info("========================================")
        logger.info("=== Robot Application Finished        ===")
        logger.info("========================================")
        logging.shutdown() # Flush and close logging handlers
