FireEye Application

Overview

The Fire Detection and Simulation Application is a machine-learning-powered system that detects fire in images and videos using the YOLOv5 model and simulates the spread of fire based on various environmental factors. The application provides a graphical user interface (GUI) using PyQt5, making it easy to analyze videos and images for fire detection, run simulations, and visualize the spread of fire over time.

Features

	•	Fire Detection: Detect fire in images and videos using the YOLOv5 object detection model.
	•	Simulation: Simulate the spread of fire over time based on input parameters such as wind speed, temperature, humidity, and slope.
	•	Database Integration: Store and retrieve videos and images from an SQLite database for analysis.
	•	Real-time Visualization: Visualize fire spread on a grid using Matplotlib.
	•	User-friendly GUI: PyQt5-based GUI for easy interaction and visualization.

Requirements

	•	Python 3.8 or later
	•	Hardware:
	•	CPU (minimum): Intel Core i5
	•	RAM: 8GB or more
	•	Optional: NVIDIA GPU for faster model inference (if using GPU-enabled Torch)

Installation

1. Clone the Repository
   git clone https://github.com/ranekhuri/fire_detection
   
2. Install Dependencies

Use the requirements.txt file to install all the necessary Python packages:
  pip install -r requirements.txt
  
3. Set Up YOLOv5 Model

Download the YOLOv5 model and place the trained weights file (best.pt) in the appropriate directory:
	  1.	Clone the YOLOv5 repository:
        git clone https://github.com/ultralytics/yolov5
        cd yolov5
    2.	Place the model weights (best.pt) inside the yolov5/runs/train/exp5/weights/ directory.

4. Run the Application

To start the application, run:
python main.py

Usage

	1.	Select Database: Start by selecting an existing database or creating a new one using the GUI. This will store all your video/image data.
	2.	Upload Data: Upload images and videos to the database for later analysis.
	3.	Process Data: Select a video or image from the database and analyze it for fire detection.
	4.	Run Simulation: Configure environmental parameters (wind speed, temperature, humidity) and run a fire spread simulation.
	5.	View Results: Visualize both the fire detection results and the fire spread simulation in the GUI.

Key Modules

	1.	MainWindow: The main window that provides navigation between different application features.
	2.	SelectDatabaseWindow: Allows the user to select or create an SQLite database for storing images, videos, and fire simulation data.
	3.	UploadDataWindow: Enables users to upload videos and images into the database.
	4.	ProcessVideoWindow: Allows users to process videos/images from the database for fire detection and displays the results.
	5.	SelectEnvironmentParametersWindow: Allows the user to set environmental parameters such as wind speed, temperature, and humidity for fire simulations.
	6.	RunSimulationWindow: Simulates the spread of fire based on the user’s input and visualizes the results.

How to Update the YOLOv5 Model

If you need to update the YOLOv5 model:

	1.	Download the latest version of the best.pt weights file or train a new one.
	2.	Replace the old weights in the yolov5/runs/train/exp5/weights/ directory with the new file.
	3.	Test the new model with sample data to ensure accuracy.

Troubleshooting

	1.	Model Not Loading: If the YOLOv5 model fails to load, ensure the correct path to the best.pt file is specified.
	2.	Slow Performance: Consider enabling GPU support by installing CUDA-enabled versions of Torch (torch==1.9.0+cu111).
	3.	Database Issues: Ensure the SQLite database is properly connected, and verify the database file exists and is not corrupted.

Contributing

We welcome contributions! If you want to improve this project, feel free to fork the repository and create a pull request.
