import logging
import time
import random
import numpy as np
import tensorflow as tf
try:
    import tensorflow_federated as tff
    TFF_AVAILABLE = True
except ImportError:
    TFF_AVAILABLE = False
    print("Warning: tensorflow_federated not installed. FL client simulation will be limited.")

from collections import OrderedDict # Useful for metrics

class FederatedLearningClient:
    """
    Placeholder client simulating participation in Federated Learning
    using TensorFlow Federated (TFF) concepts to update a local Keras model.
    """
    def __init__(self, keras_model, fl_server_url="http://placeholder.fl.server"):
        self.logger = logging.getLogger(__name__)
        self.local_model = keras_model # The actual Keras model instance to be trained
        self.server_url = fl_server_url
        self.tff_available = TFF_AVAILABLE
        if not self.tff_available:
            self.logger.warning("TFF library not found. FL functionality is simulated without TFF specifics.")
        self.logger.info("Federated Learning Client initialized for server: %s. TFF Available: %s",
                         self.server_url, self.tff_available)
        # Store initial weights shape/structure for delta calculation
        try:
             self.initial_weights = [w.numpy() for w in self.local_model.weights]
             self.logger.debug("Stored initial model weight structure for FL.")
        except Exception as e:
             self.logger.error("Could not get initial model weights: %s", e, exc_info=True)
             self.initial_weights = None

    def _prepare_tf_dataset(self, local_data_batch: list, batch_size=16) -> tf.data.Dataset | None:
        """
        Converts the collected local data batch into a tf.data.Dataset suitable for training.
        Assumes local_data_batch is a list of sequences, where each sequence represents history.
        Needs to create (input_sequence, target_sequence) pairs.
        """
        # TODO: Implement data conversion based on ObjectBehaviorLearner's input/output needs.
        # Example: Assuming each item in local_data_batch is a sequence of states [S1, S2, ..., Sn]
        # And the Transformer predicts the *next* state for each input state.
        # Input sequence: [S1, S2, ..., Sn-1], Target sequence: [S2, S3, ..., Sn]
        self.logger.debug("Preparing local data (%d sequences) for TF Dataset...", len(local_data_batch))
        input_sequences = []
        target_sequences = []
        min_len_for_pair = 2 # Need at least 2 states to form a pair

        for sequence in local_data_batch:
            if len(sequence) >= min_len_for_pair:
                input_sequences.append(sequence[:-1]) # All but last state
                target_sequences.append(sequence[1:]) # All but first state

        if not input_sequences:
            self.logger.warning("Not enough valid sequences in local data batch to create training dataset.")
            return None

        try:
            # Convert lists of numpy arrays to tensors
            input_tensor = tf.ragged.constant(input_sequences, dtype=tf.float32)
            target_tensor = tf.ragged.constant(target_sequences, dtype=tf.float32)

            # Create dataset (handle potential raggedness if sequences vary slightly)
            dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor))
            # Batch the dataset
            # Use padded_batch if sequences have variable lengths after filtering
            dataset = dataset.shuffle(buffer_size=len(input_sequences)).batch(batch_size)
            # dataset = dataset.shuffle(buffer_size=len(input_sequences)).padded_batch(batch_size) # If ragged needed
            self.logger.debug("TF Dataset prepared successfully.")
            return dataset
        except Exception as e:
            self.logger.error("Failed to create TF Dataset from local data: %s", e, exc_info=True)
            return None

    def check_participation_and_get_config(self) -> dict | None:
        """Simulates checking FL server if participation is requested and getting config."""
        self.logger.debug("Checking FL server for participation request...")
        # TODO: Implement actual HTTP GET request to self.server_url/participate
        participate = random.random() < 0.1 # Simulate 10% chance
        if participate:
            config = {
                "participate": True,
                "model_version": f"v1.{random.randint(0,9)}.{random.randint(0,9)}", # Simulate version
                "epochs": 1, # Number of local epochs
                "batch_size": 16,
                # Add other hyperparameters if needed (learning rate, etc.)
            }
            self.logger.info("Selected for FL training round: Config=%s", config)
            return config
        else:
            # self.logger.debug("Not selected for this FL round.")
            return None # Return None if not participating

    def get_server_model_weights(self, config: dict) -> list | None:
        """Simulates fetching the current global model weights from the server."""
        model_version = config.get("model_version", "unknown")
        self.logger.info("Fetching global model weights (version %s) from FL server...", model_version)
        # TODO: Implement actual HTTP GET request to self.server_url/get_weights?version=...
        # Simulate success
        if self.initial_weights is not None:
            self.logger.info("Successfully fetched global model weights (simulated).")
            # In reality, deserialize weights received from server
            # For simulation, just return the structure we stored, maybe slightly modified
            simulated_weights = [w + np.random.normal(0, 0.01, size=w.shape).astype(w.dtype) for w in self.initial_weights]
            return simulated_weights
        else:
            self.logger.error("Cannot simulate fetching weights, initial weights not stored.")
            return None

    def run_local_training_and_compute_update(self, server_weights: list, train_dataset: tf.data.Dataset, config: dict) -> dict | None:
        """
        Sets local model weights, performs local training, computes model delta.
        """
        if server_weights is None or train_dataset is None:
            self.logger.error("Cannot run local training: Missing server weights or training data.")
            return None

        epochs = config.get("epochs", 1)
        self.logger.info("Running local FL training for %d epochs...", epochs)

        try:
            # 1. Set local model weights to server weights
            # Ensure weight structure matches - TFF handles this more formally
            if len(self.local_model.weights) != len(server_weights):
                 self.logger.error("Server weights structure mismatch with local model.")
                 return None
            self.local_model.set_weights(server_weights)
            initial_weights_for_delta = [w.numpy() for w in self.local_model.weights] # Copy weights *before* training

            # 2. Perform local training
            # Use model.fit - TFF typically uses custom TF loops via tff.tf_computation
            history = self.local_model.fit(train_dataset, epochs=epochs, verbose=0)
            final_loss = history.history['loss'][-1]
            self.logger.info("Local training finished. Final loss: %.4f", final_loss)

            # 3. Compute model delta (difference between final and initial weights)
            final_weights = [w.numpy() for w in self.local_model.weights]
            weight_delta = [final - initial for final, initial in zip(final_weights, initial_weights_for_delta)]

            # 4. Prepare results (client update)
            client_update = {
                'weights_delta': weight_delta,
                'num_samples': tf.data.experimental.cardinality(train_dataset).numpy() * config.get("batch_size", 16), # Approx samples
                'metrics': OrderedDict([('loss', final_loss)]) # Example metric
            }
            # TFF formally defines ClientOutput structure

            return client_update

        except Exception as e:
            self.logger.error("Error during local training or delta computation: %s", e, exc_info=True)
            return None


    def send_updates(self, client_update: dict) -> bool:
        """Simulates sending model updates (weight delta) back to the FL server."""
        if client_update is None: return False
        num_layers = len(client_update.get('weights_delta', []))
        num_samples = client_update.get('num_samples', 0)
        self.logger.info("Sending FL updates (delta for %d layers, %d samples) to server...", num_layers, num_samples)
        # TODO: Implement secure HTTP POST of client_update (serialized) to self.server_url/submit_update
        success = random.random() < 0.95 # Simulate success
        if success:
             self.logger.info("Successfully sent FL updates.")
             return True
        else:
             self.logger.error("Simulated failure sending FL updates.")
             return False

    def run_fl_round(self, local_data_batch: list):
        """
        Orchestrates one potential round of federated learning participation.
        """
        if not self.tff_available and self.initial_weights is None:
             self.logger.warning("Skipping FL round simulation: TFF not available or initial weights missing.")
             return

        # 1. Check for participation
        config = self.check_participation_and_get_config()
        if not config:
             # Not participating this round
             # Optionally check for new global model periodically even if not training
             # if random.random() < 0.1: self.apply_server_model_weights() # Example
             return

        # 2. Prepare data
        train_dataset = self._prepare_tf_dataset(local_data_batch, config.get("batch_size", 16))
        if train_dataset is None:
             self.logger.error("FL round aborted: Failed to prepare local dataset.")
             return

        # 3. Get server model
        server_weights = self.get_server_model_weights(config)
        if server_weights is None:
             self.logger.error("FL round aborted: Failed to get server model weights.")
             return

        # 4. Train locally and compute updates
        client_update = self.run_local_training_and_compute_update(server_weights, train_dataset, config)
        if client_update is None:
             self.logger.error("FL round aborted: Local training or update computation failed.")
             return

        # 5. Send updates to server
        self.send_updates(client_update)

        # Note: In real TFF, applying the *next* global model usually happens
        # at the start of the *next* round when get_server_model_weights is called.
        # The local model state after training isn't necessarily kept unless specified.
