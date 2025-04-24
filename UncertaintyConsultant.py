import logging
import time
import numpy as np
# Assuming BNNObjectBehaviorLearner is defined in the same file or imported
# from your_module import BNNObjectBehaviorLearner

# Define necessary constants (adjust these based on experimentation)
HIGH_UNCERTAINTY_THRESHOLD = 0.5 # Example variance threshold
PLANNER_SAFETY_FACTOR = 1.5    # Example factor to scale margin by std dev (sqrt(var))

class UncertaintyConsultant:
    """
    Manages BNN models to provide uncertainty estimates for robot behavior modulation.
    """
    def __init__(self, bnn_learner_instance: BNNObjectBehaviorLearner, update_frequency_hz=1.0):
        """
        Args:
            bnn_learner_instance: An instantiated BNNObjectBehaviorLearner.
            update_frequency_hz: How often (approximately) to run BNN analysis.
        """
        self.logger = logging.getLogger(__name__)
        if not isinstance(bnn_learner_instance, BNNObjectBehaviorLearner):
             self.logger.error("UncertaintyConsultant requires a BNNObjectBehaviorLearner instance.")
             raise TypeError("Invalid bnn_learner_instance type.")

        self.bnn_learner = bnn_learner_instance
        self.current_uncertainty_metrics = {} # Store latest metrics
        self.update_period_secs = 1.0 / update_frequency_hz if update_frequency_hz > 0 else float('inf')
        self.last_update_time = 0
        self.logger.info("UncertaintyConsultant initialized with update period %.2f s.", self.update_period_secs)

    def update_uncertainty(self, object_state_history: dict):
        """
        Runs BNN prediction periodically and updates internal uncertainty metrics.

        Args:
            object_state_history (dict): {object_id: deque(maxlen=history_length)}
                                         containing recent state sequences.
        """
        current_time = time.time()
        if current_time - self.last_update_time < self.update_period_secs:
            return # Update less frequently

        self.logger.debug("Running BNN uncertainty analysis...")
        overall_variance_sum = 0.0
        num_predictions = 0
        high_uncertainty_objects = []
        object_variances = {} # Store per-object variance metric

        for obj_id, history_deque in object_state_history.items():
            if len(history_deque) == history_deque.maxlen: # Ensure full history
                history_array = np.array(list(history_deque)).reshape(1, history_deque.maxlen, self.bnn_learner.state_dimension)
                try:
                    # Use the BNN predict method which returns mean and variance
                    mean_pred, var_pred = self.bnn_learner.predict(history_array)

                    if var_pred is not None:
                        # Calculate a representative metric from variance tensor
                        # Example: Average variance over prediction horizon for position (x,y)
                        # Assumes state is [x, y, vx, vy]
                        pos_variance = var_pred[0, :, :2] # Shape: (time_steps, 2) for x,y variance
                        avg_pos_var = tf.reduce_mean(pos_variance).numpy() # Avg variance for x,y over time

                        overall_variance_sum += avg_pos_var
                        num_predictions += 1
                        object_variances[obj_id] = avg_pos_var

                        if avg_pos_var > HIGH_UNCERTAINTY_THRESHOLD:
                            high_uncertainty_objects.append(obj_id)
                            self.logger.debug("Object %s has high positional variance: %.4f", obj_id, avg_pos_var)

                except Exception as e:
                    self.logger.error("Error during BNN prediction for object %s: %s", obj_id, e, exc_info=True)
            # else: Skip objects without full history

        avg_overall_variance = overall_variance_sum / num_predictions if num_predictions > 0 else 0.0

        # Store the calculated metrics
        self.current_uncertainty_metrics = {
            "average_positional_variance": avg_overall_variance,
            "high_uncertainty_ids": high_uncertainty_objects,
            "object_variances": object_variances # Store per-object metric
        }
        self.last_update_time = current_time
        self.logger.info("Uncertainty metrics updated: AvgVar=%.4f, HighUncertainty=%d objects",
                         avg_overall_variance, len(high_uncertainty_objects))
        self.logger.debug("Detailed metrics: %s", self.current_uncertainty_metrics)


    def get_metrics(self) -> dict:
        """Provides the latest calculated uncertainty metrics."""
        return self.current_uncertainty_metrics
