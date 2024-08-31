# Multi-Model Object Tracking and Prediction for Autonomous Exploration

This repository contains Python code for advanced object tracking and prediction in an autonomous exploration scenario. It integrates multiple machine learning models, including object detection (e.g., YOLO), Kalman filters, Transformers, and Graph Neural Networks (GNNs), to learn and anticipate object behavior, ultimately guiding a robot's exploration path.

## Core Components:

* **Object Detection:** Employs a pre-trained object detection model (e.g., YOLO) to identify and locate objects in the robot's environment.
* **Kalman Filtering:** Implements individual Kalman filters for each detected object to track their movement and estimate future positions.
* **Transformer Model:** A Transformer-based neural network learns the dynamic behavior of objects over time based on their state histories.
* **Graph Neural Network:** A GNN model leverages the relationships between objects (represented as a graph) to refine predictions and account for potential interactions.
* **Prediction Combiner:**  Combines predictions from the Transformer and GNN models, taking into account their confidence levels and contextual information from the environment and robot state.
* **Environment Information Node:** A ROS node to subscribe to sensor data (e.g., Lidar) and provide contextual information for the models.
* **Google Cloud Pub/Sub:**  Utilizes Google Cloud Pub/Sub for communication and coordination between multiple robots or nodes in a distributed system.

## Dependencies:

* **Python:** Ensure you have a compatible version of Python installed (e.g., Python 3.7 or higher).
* **Libraries:** Install the necessary libraries using `pip`:
    ```bash
    pip install numpy opencv-python tensorflow networkx torch torch-geometric rclpy sensor-msgs google-cloud-pubsub google-cloud-aiplatform
    ```
* **Object Detection Model:** Download and configure a pre-trained object detection model compatible with OpenCV (e.g., YOLOv3).

## Running the Code:

1. **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    ```

2. **Set up your environment:**
    * Configure your object detection model (e.g., YOLO) and provide its configuration and weight files in the code.
    * Set up your Google Cloud project and create Pub/Sub topics and subscriptions as needed.

3. **Run the main script:**
    ```bash
    python your_main_script.py
    ```

## Important Considerations:

* **Pre-trained Models:** This code assumes you have a pre-trained object detection model ready. Ensure it's compatible with OpenCV.
* **Customization:** Adapt the code to your specific robot platform and sensor configuration.
* **Distributed Setup:** If running multiple robots, configure Pub/Sub communication appropriately.
* **Exploration Logic:** Implement the `object_is_unexplored`, `object_confidence_is_low`, `query_gemini`, `update_object_graph`, `should_explore_object`, and `explore_object` functions to define your exploration strategy based on object properties and Gemini responses.
* **Data Collection:** If you have pre-training data, implement the `load_pre_training_data` function to load it.
* **Environment and Robot State:**  Implement the `get_environment_info` and `get_robot_state` functions to retrieve relevant contextual data.
* **Path Planning:** Implement the `smooth_path` and `execute_path` functions to generate and execute smooth paths for the robot.

## Contributing:

Contributions are welcome! Feel free to open issues or submit pull requests to enhance this code.

## License:

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments:

* This code was inspired by research in multi-model object tracking and prediction for autonomous systems.
* We thank the developers of the libraries used in this project.
