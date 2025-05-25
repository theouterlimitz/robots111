# Project Overview

This project is a sophisticated robotics system designed for autonomous navigation and interaction in dynamic environments.

## Key Capabilities

- Intelligent object perception and behavior prediction.
- Adaptive navigation using advanced path planning.
- Collaborative learning through federated methods.
- Cloud-integrated data management and processing.

## Features

- Object detection and tracking (YOLO, Kalman Filter)
- Bayesian Neural Network (BNN) for behavior prediction with uncertainty estimation
- Federated learning for distributed model training
- RRT* path planning for navigation in dynamic environments
- Cloud integration (Google Cloud Storage, Google Cloud Functions for graph pruning)
- ROS2 integration for sensor data (Lidar) and robot control
- Natural language command understanding (via Google Gemini API)
- Uncertainty-aware navigation through an Uncertainty Consultant module

## Components

-   **`BNN.py`**: Implements a Bayesian Neural Network (BNN) using TensorFlow and Keras. It's designed for object behavior learning and prediction, incorporating Monte Carlo Dropout for uncertainty estimation.
-   **`FederatedLearningClient.py`**: Contains the client-side logic for federated learning. It allows the BNN model to be trained across decentralized datasets without sharing raw data, using TensorFlow Federated (TFF) concepts.
-   **`UncertaintyConsultant.py`**: Manages and utilizes the BNN model to provide real-time uncertainty estimates. These metrics are used to modulate robot behavior, making navigation safer and more adaptive.
-   **`UnitTest.py`**: Includes unit tests for various components of the system, ensuring reliability and correctness.
-   **`cloud-function.py`**: A Google Cloud Function designed to be triggered periodically (e.g., by Cloud Scheduler). Its purpose is to prune stale nodes from an object graph stored in Google Cloud Storage (GCS), maintaining data relevance.
-   **`path planner.py`**: Implements the RRT* (Rapidly-exploring Random Tree Star) path planning algorithm. This allows the robot to find optimal paths in complex and dynamic environments, considering static and predicted dynamic obstacles.
-   **`robots.py`**: This is the main application file that integrates all other components. It handles robot control, perception (object detection via YOLO, Lidar processing via ROS2), state estimation (Kalman Filters), behavior prediction (using BNN and UncertaintyConsultant), navigation (using RRT* planner and Gemini API for high-level commands), and communication with cloud services (GCS, Pub/Sub, Neptune API placeholder).

## Setup and Usage

### Dependencies

Ensure you have the following dependencies installed:

-   **Python**: Python 3.8+ is recommended.
-   **TensorFlow**: For BNN and other ML models. (`pip install tensorflow`)
-   **TensorFlow Federated**: For federated learning. (`pip install tensorflow_federated`)
-   **OpenCV (cv2)**: For object detection and Kalman filters. (`pip install opencv-python`)
-   **NumPy**: For numerical operations. (`pip install numpy`)
-   **NetworkX**: For graph operations. (`pip install networkx`)
-   **Google Cloud Client Libraries**:
    *   `pip install google-cloud-storage`
    *   `pip install google-cloud-pubsub`
    *   `pip install google-generativeai` (for Gemini API)
-   **ROS2**: Installation depends on your OS and desired distribution (e.g., Foxy, Galactic, Humble). Please refer to the [official ROS2 documentation](https://docs.ros.org/en/humble/Installation.html) for installation instructions.
-   **SciPy**: For KDTree in RRT* and other scientific computations. (`pip install scipy`)

It is recommended to use a virtual environment (e.g., `venv` or `conda`) to manage project dependencies.

### Configuration

1.  **API Keys & Environment Variables**:
    *   Set the `GEMINI_API_KEY` environment variable with your Google Gemini API key.
    *   Ensure other necessary API keys or service account credentials for Google Cloud are configured, typically via the `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to your service account JSON file.

2.  **Model Paths**:
    *   Configure paths to YOLO model configuration (`.cfg`) and weights (`.weights`) files within `robots.py` or via environment variables if adapted (e.g., `YOLO_CFG_PATH`, `YOLO_WEIGHTS_PATH`).

3.  **Google Cloud Project**:
    *   Set your `GCP_PROJECT_ID` environment variable.
    *   Configure necessary Google Cloud Storage bucket names and Pub/Sub topic IDs within the relevant scripts (`robots.py`, `cloud-function.py`).

### Running the System

1.  **Main Robot Application (`robots.py`)**:
    *   Ensure your ROS2 environment is sourced if you are using ROS2 for sensor input or robot control. This might involve a command like `source /opt/ros/<your_ros_distro>/setup.bash` or `source install/setup.bash` if you built ROS2 packages from source.
    *   Execute the main script: `python robots.py`
    *   The script may require specific command-line arguments or further configuration depending on the operational mode.

2.  **Cloud Function (`cloud-function.py`)**:
    *   This function is intended for deployment to Google Cloud Functions.
    *   You will need to use the `gcloud` command-line tool to deploy it, specifying the trigger (e.g., Pub/Sub, HTTP, Cloud Scheduler), runtime (Python), and necessary environment variables. Refer to the [Google Cloud Functions documentation](https://cloud.google.com/functions/docs/deploying) for deployment instructions.

3.  **Unit Tests (`UnitTest.py`)**:
    *   Run the unit tests to ensure components are functioning correctly: `python UnitTest.py`

This provides a general guide. Specific setup steps might vary based on your exact environment and the features you intend to use. Refer to individual script comments and code for more detailed configuration.

## Testing

The project includes unit tests located in `UnitTest.py`. These tests are designed to verify the correctness and reliability of individual components and functionalities.

To run the tests, execute the following command in your terminal:

```bash
python UnitTest.py
```

The test environment generally uses the same dependencies as the main application. Ensure all required libraries listed in the "Dependencies" section are installed.

**Note**: The current set of tests in `UnitTest.py` might be placeholders or incomplete for some of the more complex components. Contributions to expand test coverage are welcome. Please refer to the "Contributing" section for more details on how to contribute.

## Contributing

We welcome contributions to enhance and expand this project! Whether it's adding new features, fixing bugs, improving documentation, or writing more tests, your help is appreciated.

### Areas for Contribution

-   **New Features**: Implementing new capabilities for the robot, such as advanced sensor fusion, new behaviors, or improved human-robot interaction.
-   **Bug Fixes**: Identifying and resolving issues in the existing codebase.
-   **Documentation**: Improving the clarity and completeness of this README, code comments, or other documentation.
-   **Testing**: Expanding the unit test suite (`UnitTest.py`) or adding integration tests to ensure robustness.
-   **Performance Optimizations**: Enhancing the efficiency of algorithms or data processing.

### Contribution Workflow

We follow a standard fork-and-pull-request workflow:

1.  **Fork the Repository**: Create your own fork of the main repository on GitHub.
2.  **Create a Branch**: Before making changes, create a new branch in your fork. Use a descriptive name, such as:
    *   `git checkout -b feature/your-new-feature-name`
    *   `git checkout -b bugfix/issue-tracker-number`
3.  **Make Your Changes**: Implement your feature or fix in your branch.
4.  **Adhere to Style Guidelines**:
    *   For Python code, please follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines.
    *   Ensure your code is well-commented where necessary.
5.  **Add/Update Tests**:
    *   If you're adding a new feature, please include new tests for it.
    *   If you're fixing a bug, add a test that reproduces the bug and verifies your fix.
6.  **Commit Your Changes**: Make clear and descriptive commit messages. This helps in understanding the history of changes.
    ```bash
    git commit -m "feat: Implement X feature"
    git commit -m "fix: Resolve Y bug in Z component"
    ```
7.  **Push to Your Fork**: Push your committed changes to your branch on your forked repository.
    ```bash
    git push origin feature/your-new-feature-name
    ```
8.  **Submit a Pull Request (PR)**: Open a pull request from your branch in your fork to the `main` branch (or the relevant target branch) of the original repository.
    *   Provide a clear title and description for your PR, explaining the changes and referencing any related issues.

### Discussing Significant Changes

For substantial changes, such as adding a major new component or significantly refactoring existing code, please **open an issue first** to discuss your ideas. This helps ensure that your proposed changes align with the project's goals and that your efforts are not wasted.

Thank you for considering contributing to this project!

## License

This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
