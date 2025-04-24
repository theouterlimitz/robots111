import logging
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, TimeDistributed,
                                     MultiHeadAttention, LayerNormalization, Dropout)
from tensorflow.keras.models import Model

# Assuming TransformerBlock and ObjectBehaviorLearner are defined as before
# Need to ensure they are accessible (e.g., defined in the same file or imported)

# Re-define base classes here for completeness in this code block:
# (Keep definitions from previous versions, including logging and error handling)

class TransformerBlock(tf.keras.layers.Layer):
    # ... (Full implementation from previous step) ...
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.mha = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model // self.num_heads if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads == 0 else self.d_model)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential(
            [Dense(self.dff, activation='relu'), Dense(self.d_model)]
        )
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        # Ensure dropout layers exist and have a non-zero rate for MC Dropout
        self.dropout1 = Dropout(self.rate)
        self.dropout2 = Dropout(self.rate)

    def call(self, x, training, mask=None): # Pass training flag
        # Need to ensure the 'training' flag propagates to Dropout layers
        attn_output, attn_weights = self.mha(query=x, value=x, key=x, attention_mask=mask, return_attention_scores=True, training=training)
        attn_output = self.dropout1(attn_output, training=training) # Pass training flag
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training) # Pass training flag to Sequential if needed
        ffn_output = self.dropout2(ffn_output, training=training) # Pass training flag
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

class ObjectBehaviorLearner:
    # ... (Keep __init__, train, update_local_cache from previous version) ...
    def __init__(self, prediction_horizon=10, state_dimension=4, d_model=128, num_heads=8, dff=512, num_layers=2):
        self.logger = logging.getLogger(__name__)
        self.prediction_horizon = prediction_horizon
        self.state_dimension = state_dimension
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.num_layers = num_layers
        self.local_object_graph_cache = nx.Graph() # Example cache
        self.logger.info("ObjectBehaviorLearner initialized.")
        # Build the Keras model
        try:
            if self.d_model > 0 and self.num_heads > 0 and self.d_model % self.num_heads != 0:
                 self.logger.warning("Transformer d_model (%d) is not divisible by num_heads (%d).", self.d_model, self.num_heads)

            input_layer = Input(shape=(None, self.state_dimension), name="transformer_input")
            # Embedding layer
            x = Dense(d_model, name="input_embedding")(input_layer)
            # Positional Encoding would go here if used

            # Transformer Blocks - IMPORTANT: Need to pass the 'training' argument through
            # We define the layers here, but the training flag is passed during call
            self.transformer_blocks = [TransformerBlock(d_model, num_heads, dff, name=f"transformer_block_{i}") for i in range(num_layers)]
            self.output_dense_layer = TimeDistributed(Dense(self.state_dimension), name="output_dense")

            # Define the forward pass logic using a functional model or subclassing Model
            # Using subclassing here for clarity on passing the training flag
            class TransformerModel(tf.keras.Model):
                def __init__(self, embedding_layer, transformer_blocks, output_dense_layer, **kwargs):
                    super().__init__(**kwargs)
                    self.embedding_layer = embedding_layer
                    self.transformer_blocks = transformer_blocks
                    self.output_dense_layer = output_dense_layer

                def call(self, inputs, training=None, mask=None):
                    x = self.embedding_layer(inputs)
                    # Add positional encoding here if needed
                    for block in self.transformer_blocks:
                        x = block(x, training=training, mask=mask) # Pass training flag here
                    return self.output_dense_layer(x)

            # Instantiate the inner model
            self.model = TransformerModel(
                embedding_layer=Dense(d_model, name="input_embedding"),
                transformer_blocks=self.transformer_blocks,
                output_dense_layer=self.output_dense_layer
            )
            # Build the model by calling it once (optional but good practice)
            # Need dummy input matching expected shape (batch, time, state_dim)
            dummy_input_shape = (1, prediction_horizon if prediction_horizon else 10, state_dimension) # Example shape
            self.model.build(input_shape=dummy_input_shape)

            self.model.compile(optimizer='adam', loss='mse')
            self.logger.info("Transformer Model built and compiled successfully.")
            # self.model.summary(print_fn=self.logger.info)

        except Exception as e:
            self.logger.critical("Failed to build or compile Transformer model: %s", e, exc_info=True)
            raise

    def train(self, past_states, target_states):
        # (Keep implementation)
        self.logger.info("Starting Transformer training...")
        try:
            # Pass training=True implicitly via fit
            history = self.model.fit(past_states, target_states, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            self.logger.info("Transformer training finished. Final validation loss: %s", history.history['val_loss'][-1])
        except Exception as e:
            self.logger.error("Error during Transformer training: %s", e, exc_info=True)


    def predict(self, past_states):
        """Standard prediction (point estimate) without uncertainty."""
        self.logger.debug("Standard prediction for %d samples", tf.shape(past_states)[0])
        try:
            # Pass training=False for standard prediction to disable dropout
            predictions = self.model(past_states, training=False)
            # predictions = self.model.predict(past_states) # .predict also sets training=False
            return predictions
        except Exception as e:
             self.logger.error("Error during standard Transformer prediction: %s", e, exc_info=True)
             return None

    def update_local_cache(self, object_id, state, timestamp_iso):
        # (Keep implementation)
        pass # See previous versions for implementation

# --- BNN Variant using MC Dropout ---

class BNNObjectBehaviorLearner(ObjectBehaviorLearner):
    """
    BNN version of ObjectBehaviorLearner using MC Dropout for uncertainty estimation.
    Inherits model structure and training from ObjectBehaviorLearner.
    Overrides the predict method to perform MC Dropout inference.
    """
    def __init__(self, num_mc_samples=30, **kwargs):
        """
        Initializes the BNN Learner.

        Args:
            num_mc_samples (int): Number of Monte Carlo samples (forward passes)
                                  to run for uncertainty estimation.
            **kwargs: Arguments passed to the parent ObjectBehaviorLearner __init__.
        """
        super().__init__(**kwargs) # Initialize the base class (builds the model)
        self.logger = logging.getLogger(__name__) # Ensure logger is specific
        if not isinstance(num_mc_samples, int) or num_mc_samples <= 1:
            raise ValueError("num_mc_samples must be an integer greater than 1.")
        self.num_mc_samples = num_mc_samples
        self.logger.info("BNNObjectBehaviorLearner initialized with %d MC samples.", self.num_mc_samples)

        # Verify dropout layers exist in the built model
        has_dropout = any(isinstance(layer, Dropout) for layer in self.model.layers + [l for block in self.transformer_blocks for l in block.layers])
        if not has_dropout:
             self.logger.warning("MC Dropout requires Dropout layers in the model, but none were detected automatically. Ensure Dropout layers exist and are active during training=True.")


    @tf.function # Decorate with tf.function for potential graph optimization
    def predict_mc_dropout(self, past_states):
        """Internal TF function to run multiple predictions."""
        predictions_list = []
        for _ in tf.range(self.num_mc_samples):
            # IMPORTANT: Call the model with training=True to enable dropout
            predictions = self.model(past_states, training=True)
            predictions_list.append(predictions)
        # Stack predictions along a new dimension (axis 0)
        predictions_stack = tf.stack(predictions_list, axis=0)
        return predictions_stack

    def predict(self, past_states: np.ndarray | tf.Tensor) -> tuple[tf.Tensor | None, tf.Tensor | None]:
        """
        Performs prediction using MC Dropout to estimate mean and variance.

        Args:
            past_states: Input tensor/array of shape (batch_size, time_steps, state_dimension).

        Returns:
            tuple: (mean_prediction, variance_prediction)
                   mean_prediction: Tensor shape (batch_size, time_steps, state_dimension)
                   variance_prediction: Tensor shape (batch_size, time_steps, state_dimension)
                   Returns (None, None) on error.
        """
        start_time = time.time()
        self.logger.debug("BNN MC Dropout prediction started for input shape %s...", tf.shape(past_states).numpy())

        try:
            # Ensure input is a Tensor
            if not isinstance(past_states, tf.Tensor):
                past_states = tf.constant(past_states, dtype=tf.float32)

            # Run multiple forward passes with dropout enabled
            predictions_stack = self.predict_mc_dropout(past_states)
            # Shape: (num_mc_samples, batch_size, time_steps, state_dimension)

            # Calculate mean across the samples dimension (axis 0)
            mean_prediction = tf.reduce_mean(predictions_stack, axis=0)

            # Calculate variance across the samples dimension (axis 0)
            variance_prediction = tf.math.reduce_variance(predictions_stack, axis=0)
            # Optionally calculate standard deviation: tf.math.reduce_std(predictions_stack, axis=0)

            duration = time.time() - start_time
            self.logger.debug("MC Dropout prediction finished in %.3f seconds.", duration)

            return mean_prediction, variance_prediction

        except Exception as e:
            self.logger.error("Error during MC Dropout prediction: %s", e, exc_info=True)
            return None, None

    # Inherits train and update_local_cache methods from ObjectBehaviorLearner
    # The training method of the base class is sufficient as it enables dropout naturally.
