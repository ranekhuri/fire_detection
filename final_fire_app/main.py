import copy
import sys
from PIL import Image

from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel, QSlider, \
    QDoubleSpinBox, QComboBox, QFileDialog, QMessageBox, QInputDialog, QLineEdit, QHBoxLayout, QSpacerItem, QSizePolicy

from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5.QtCore import Qt
import sqlite3
import json
import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Start measuring time
start_time = time.time()
# End measuring time
end_time = time.time()
runtime = end_time - start_time
# Calculate and display the runtime
#print(f"Video analysis runtime: {runtime:.2f} seconds")

user_selected_db = None
model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt', source='local')

wind_direction = None
wind_speed = None
temperature = None
humidity = None
slope_factor = None
vegetation = None





class SecondaryWindow(QWidget):
    def __init__(self, window_number):
        super().__init__()
        self.setWindowTitle(f"Window {window_number}")
        self.setGeometry(100, 100, 200, 100)
        self.label = QLabel(f"This is window {window_number}", self)
        self.label.move(50, 40)

class SelectDatabaseWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Database")
        self.setGeometry(200, 200, 400, 200)
        # Main layout
        layout = QVBoxLayout()

        # Label for instructions
        self.label = QLabel("Select or create a database", self)
        layout.addWidget(self.label)

        # ComboBox to select from existing databases
        self.db_combo = QComboBox(self)
        self.populate_database_list()
        layout.addWidget(self.db_combo)

        # Button to select an existing database
        self.select_button = QPushButton("Select Database", self)
        self.select_button.clicked.connect(self.select_database)
        layout.addWidget(self.select_button)

        # Button to create a new database
        self.create_button = QPushButton("Create New Database", self)
        self.create_button.clicked.connect(self.create_new_database)
        layout.addWidget(self.create_button)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)

        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)


        # Set the main layout
        self.setLayout(layout)

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    # Function to populate the ComboBox with existing database files
    def populate_database_list(self):
        databases = ['first_fire.db', 'second_fire.db', 'third_fire.db']  # Add more database file names here
        self.db_combo.addItems(databases)

    # Function to select the current database
    def select_database(self):
        selected_db = self.db_combo.currentText()
        if selected_db:
            self.init_db(selected_db)
            QMessageBox.information(self, "Database Selected", f"Connected to {selected_db}")

    # Function to create a new database
    def create_new_database(self):
        new_db_name, ok = QInputDialog.getText(self, "Create New Database", "Enter database name:")
        if ok and new_db_name:
            new_db_name = f"{new_db_name}.db" if not new_db_name.endswith('.db') else new_db_name
            self.init_db(new_db_name)
            QMessageBox.information(self, "Database Created", f"Created and connected to {new_db_name}")
            self.db_combo.addItem(new_db_name)  # Add new database to the ComboBox

    # Function to create/connect to the database and initialize tables
    def init_db(self, db_name):
        conn = sqlite3.connect(db_name)
        global user_selected_db
        user_selected_db = copy.copy(db_name)

        print('user selected '+ user_selected_db)
        c = conn.cursor()

        c.execute('''
            CREATE TABLE IF NOT EXISTS fire_simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wind_direction INTEGER,
                wind_speed REAL,
                temperature REAL,
                humidity REAL,
                slope_factor REAL,
                vegetation_type TEXT,
                simulation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                grid_data TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                image_path TEXT
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                location TEXT,
                source TEXT,
                description TEXT,
                video_path TEXT
            )
        ''')

        conn.commit()
        conn.close()



class EditVideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Edit Video Details")
        self.setGeometry(200, 200, 600, 400)

        # Main layout
        layout = QVBoxLayout()

        # Label for instructions
        self.label = QLabel("Select a video to edit its details:", self)
        layout.addWidget(self.label)

        # ComboBox to select the video
        self.video_combo = QComboBox(self)
        self.load_videos_from_database()  # Populate the combo box with videos
        layout.addWidget(self.video_combo)

        # Video player setup
        self.video_widget = QVideoWidget(self)
        layout.addWidget(self.video_widget)

        self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.media_player.setVideoOutput(self.video_widget)

        # Buttons for controlling playback
        control_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_video)
        control_layout.addWidget(self.play_button)

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_video)
        control_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_video)
        control_layout.addWidget(self.stop_button)

        layout.addLayout(control_layout)

        # Input fields for video details
        self.timestamp_edit = QLineEdit(self)
        self.timestamp_edit.setPlaceholderText("Timestamp")
        layout.addWidget(self.timestamp_edit)

        self.location_edit = QLineEdit(self)
        self.location_edit.setPlaceholderText("Location")
        layout.addWidget(self.location_edit)

        self.source_edit = QLineEdit(self)
        self.source_edit.setPlaceholderText("Source")
        layout.addWidget(self.source_edit)

        self.description_edit = QLineEdit(self)
        self.description_edit.setPlaceholderText("Description")
        layout.addWidget(self.description_edit)

        self.video_path_edit = QLineEdit(self)
        self.video_path_edit.setPlaceholderText("Video Path")
        layout.addWidget(self.video_path_edit)

        # Button to save the edited details
        self.save_button = QPushButton("Save Changes", self)
        self.save_button.clicked.connect(self.save_video_details)
        layout.addWidget(self.save_button)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)

        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)

        # Set the main layout
        self.setLayout(layout)

        # Load details of the selected video when the user changes selection
        self.video_combo.currentIndexChanged.connect(self.load_selected_video_details)


    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    # Function to load videos from the database into the ComboBox
    def load_videos_from_database(self):
        conn = sqlite3.connect('first_fire.db')  # Connect to your actual database
        cursor = conn.cursor()

        # Fetch video records
        cursor.execute("SELECT id, description FROM videos")
        videos = cursor.fetchall()

        # Populate the ComboBox with video descriptions
        self.video_combo.clear()
        for video in videos:
            self.video_combo.addItem(video[1], video[0])  # Display description, store id

        conn.close()

    # Function to load the details of the selected video
    def load_selected_video_details(self):
        video_id = self.video_combo.currentData()  # Get the video ID associated with the current selection

        if video_id:
            conn = sqlite3.connect('first_fire.db')
            cursor = conn.cursor()

            # Fetch video details by ID
            cursor.execute("SELECT timestamp, location, source, description, video_path FROM videos WHERE id = ?", (video_id,))
            video = cursor.fetchone()

            if video:
                # Fill in the input fields with the video details
                self.timestamp_edit.setText(video[0])
                self.location_edit.setText(video[1])
                self.source_edit.setText(video[2])
                self.description_edit.setText(video[3])
                self.video_path_edit.setText(video[4])

                # Load the selected video for playback
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video[4])))

            conn.close()

    # Function to save the edited video details to the database
    def save_video_details(self):
        video_id = self.video_combo.currentData()  # Get the video ID
        timestamp = self.timestamp_edit.text()
        location = self.location_edit.text()
        source = self.source_edit.text()
        description = self.description_edit.text()
        video_path = self.video_path_edit.text()


        if video_id:
            conn = sqlite3.connect('first_fire.db')
            cursor = conn.cursor()

            # Update the video details in the database
            cursor.execute('''
                UPDATE videos 
                SET timestamp = ?, location = ?, source = ?, description = ?, video_path = ?
                WHERE id = ?
            ''', (timestamp, location, source, description, video_path, video_id))

            conn.commit()
            conn.close()

            QMessageBox.information(self, "Success", "Video details updated successfully!")
        else:
            QMessageBox.warning(self, "Error", "No video selected.")

    # Functions to control video playback
    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

    def stop_video(self):
        self.media_player.stop()


class UploadDataWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Upload Data')
        self.setGeometry(100, 100, 400, 300)

        # Main layout
        layout = QVBoxLayout()

        # Label to show the selected file path
        self.file_path_label = QLabel('No file selected', self)
        layout.addWidget(self.file_path_label)

        # Button to open file dialog for images
        self.select_image_button = QPushButton('Select Image', self)
        self.select_image_button.clicked.connect(self.open_image_dialog)
        layout.addWidget(self.select_image_button)

        # Button to open file dialog for videos
        self.select_video_button = QPushButton('Select Video', self)
        self.select_video_button.clicked.connect(self.open_video_dialog)
        layout.addWidget(self.select_video_button)

        # Button to save the selected file path to the database
        self.save_button = QPushButton('Save File Path to Database', self)
        self.save_button.clicked.connect(self.save_file_path)
        layout.addWidget(self.save_button)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)

        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)


        # Set layout to the widget
        self.setLayout(layout)

        self.selected_file_path = None
        self.selected_file_type = None  # Track whether the selected file is an image or video
        print(user_selected_db)

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    def open_image_dialog(self):
        # Open file dialog to select an image
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters(["Images (*.png *.jpg *.jpeg *.bmp)"])
        if file_dialog.exec_():
            self.selected_file_path = file_dialog.selectedFiles()[0]
            self.selected_file_type = 'image'
            self.file_path_label.setText(f"Selected Image: {self.selected_file_path}")

    def open_video_dialog(self):
        # Open file dialog to select a video
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilters(["Videos (*.mp4 *.avi *.mov *.mkv)"])
        if file_dialog.exec_():
            self.selected_file_path = file_dialog.selectedFiles()[0]
            self.selected_file_type = 'video'
            self.file_path_label.setText(f"Selected Video: {self.selected_file_path}")

    def save_file_path(self):
        if self.selected_file_path and self.selected_file_type:
            if self.selected_file_type == 'image': #selected image
                description = 'Selected Fire Image'
                self.insert_image_path(description, self.selected_file_path)
            else: #selected video
                timestamp = 'now'
                location = 'israel'
                source = 'camera'
                description = 'Selected Fire video'
                self.insert_video_path(timestamp, location, source, description, self.selected_file_path)
        else:
            self.file_path_label.setText('No file selected to save.')

    # Function to insert file path into the database
    def insert_video_path(self, timestamp, location, source, description, video_path):

        if user_selected_db != None:
            print('selected db is' + user_selected_db)

            conn = sqlite3.connect(user_selected_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO videos (timestamp, location, source, description, video_path)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, location, source, description, video_path))
            conn.commit()
            conn.close()
            self.file_path_label.setText(f'{self.selected_file_type.capitalize()} path saved to database!')

        else:
            self.file_path_label.setText('No Database selected yet')

    def insert_image_path(self, description, image_path):
        if user_selected_db != None:
            print('selected db is' + user_selected_db)

            conn = sqlite3.connect(user_selected_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO images (description, image_path)
                VALUES (?, ?)
            ''', (description, image_path))
            # print('description is ' + description + 'image path is ' + image_path)
            conn.commit()
            conn.close()
            self.file_path_label.setText(f'{self.selected_file_type.capitalize()} path saved to database!')

        else:
            self.file_path_label.setText('No Database selected yet')
class SelectEnvironmentParametersWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Environment Parameters")
        self.setGeometry(200, 200, 400, 200)
        #self.label = QLabel("Select Environment Parameters", self)
        #self.label.move(50, 40)

        # Window layout
        layout = QVBoxLayout()

        # Wind Direction
        self.wind_dir_label = QLabel("Wind Direction (degrees): 270", self)
        self.wind_dir_slider = QSlider(Qt.Horizontal, self)
        self.wind_dir_slider.setRange(0, 360)
        self.wind_dir_slider.setValue(270)
        self.wind_dir_slider.valueChanged.connect(self.update_wind_direction)

        layout.addWidget(self.wind_dir_label)
        layout.addWidget(self.wind_dir_slider)

        # Wind Speed
        self.wind_speed_label = QLabel("Wind Speed (m/s): 1.5", self)
        self.wind_speed_spinbox = QDoubleSpinBox(self)  # Use QDoubleSpinBox for float values
        self.wind_speed_spinbox.setRange(0, 10)
        self.wind_speed_spinbox.setSingleStep(0.1)  # Now it accepts float values
        self.wind_speed_spinbox.setValue(1.5)
        self.wind_speed_spinbox.valueChanged.connect(self.update_wind_speed)

        layout.addWidget(self.wind_speed_label)
        layout.addWidget(self.wind_speed_spinbox)

        # Temperature
        self.temp_label = QLabel("Temperature (°C): 30", self)
        self.temp_slider = QSlider(Qt.Horizontal, self)
        self.temp_slider.setRange(0, 50)
        self.temp_slider.setValue(30)
        self.temp_slider.valueChanged.connect(self.update_temperature)

        layout.addWidget(self.temp_label)
        layout.addWidget(self.temp_slider)

        # Humidity
        self.humidity_label = QLabel("Humidity (%): 20", self)
        self.humidity_slider = QSlider(Qt.Horizontal, self)
        self.humidity_slider.setRange(0, 100)
        self.humidity_slider.setValue(20)
        self.humidity_slider.valueChanged.connect(self.update_humidity)

        layout.addWidget(self.humidity_label)
        layout.addWidget(self.humidity_slider)

        # Slope Factor
        self.slope_label = QLabel("Slope Factor: 1.2", self)
        self.slope_spinbox = QDoubleSpinBox(self)  # Use QDoubleSpinBox for float values
        self.slope_spinbox.setRange(1, 5)
        self.slope_spinbox.setSingleStep(0.1)  # Now it accepts float values
        self.slope_spinbox.setValue(1.2)
        self.slope_spinbox.valueChanged.connect(self.update_slope)

        layout.addWidget(self.slope_label)
        layout.addWidget(self.slope_spinbox)

        # Vegetation
        self.veg_label = QLabel("Vegetation Fuel Load: Medium", self)
        self.veg_combo = QComboBox(self)
        self.veg_combo.addItems(["Low", "Medium", "High"])
        self.veg_combo.currentIndexChanged.connect(self.update_vegetation)

        layout.addWidget(self.veg_label)
        layout.addWidget(self.veg_combo)

        # Run simulation button
        self.save_button = QPushButton("Save", self)
        self.save_button.clicked.connect(self.save_parameters)
        layout.addWidget(self.save_button)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)
        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)

        self.setLayout(layout)
        self.setWindowTitle('Fire Spread Simulation')

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    def update_wind_direction(self, value):
        self.wind_dir_label.setText(f"Wind Direction (degrees): {value}")

    def update_wind_speed(self, value):
        self.wind_speed_label.setText(f"Wind Speed (m/s): {value}")

    def update_temperature(self, value):
        self.temp_label.setText(f"Temperature (°C): {value}")

    def update_humidity(self, value):
        self.humidity_label.setText(f"Humidity (%): {value}")

    def update_slope(self, value):
        self.slope_label.setText(f"Slope Factor: {value}")

    def update_vegetation(self, index):
        veg_types = ["Low", "Medium", "High"]
        self.veg_label.setText(f"Vegetation Fuel Load: {veg_types[index]}")

    def save_parameters(self):
        global wind_direction
        global wind_speed
        global temperature
        global humidity
        global slope_factor
        global vegetation
        wind_direction = self.wind_dir_slider.value()
        wind_speed = self.wind_speed_spinbox.value()
        temperature = self.temp_slider.value()
        humidity = self.humidity_slider.value()
        slope_factor = self.slope_spinbox.value()
        vegetation = self.veg_combo.currentText()

        # Print parameters for now (you can integrate this into the simulation logic)
        print(f"Running simulation with parameters:\n"
              f"Wind Direction: {wind_direction} degrees\n"
              f"Wind Speed: {wind_speed} m/s\n"
              f"Temperature: {temperature} °C\n"
              f"Humidity: {humidity}%\n"
              f"Slope Factor: {slope_factor}\n"
              f"Vegetation: {vegetation}")



class ProcessVideoWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Process Data for Fire Detection")
        self.setGeometry(200, 200, 600, 600)

        # Main layout
        layout = QVBoxLayout()

        # Label for instructions
        self.label = QLabel("Select a video or image for fire detection:", self)
        layout.addWidget(self.label)

        # ComboBox to select video from database
        self.video_combo = QComboBox(self)
        self.load_videos_from_database()
        layout.addWidget(self.video_combo)

        # Button to start video analysis
        self.analyze_video_button = QPushButton("Analyze Selected Video", self)
        self.analyze_video_button.clicked.connect(self.analyze_video)
        layout.addWidget(self.analyze_video_button)

        # ComboBox to select image from database
        self.image_combo = QComboBox(self)
        self.load_images_from_database()
        layout.addWidget(self.image_combo)

        # Button to start image analysis
        self.analyze_image_button = QPushButton("Analyze Selected Image", self)
        self.analyze_image_button.clicked.connect(self.analyze_image)
        layout.addWidget(self.analyze_image_button)

        # Label to show detection results
        self.result_label = QLabel("", self)
        layout.addWidget(self.result_label)

        # Label to show the detected frame/image
        self.frame_label = QLabel(self)
        self.frame_label.setFixedSize(640, 480)  # Set the size for the QLabel
        layout.addWidget(self.frame_label)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)
        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)
        # Add the help layout to the main layout
        layout.addLayout(help_layout)

        # Set layout
        self.setLayout(layout)

        # Load the fire detection model
        #self.model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt', source='local')

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    # Function to load videos from the database into the ComboBox
    def load_videos_from_database(self):
        conn = sqlite3.connect('first_fire.db')  # Connect to your actual database
        cursor = conn.cursor()

        # Fetch video records
        cursor.execute("SELECT id, description, video_path FROM videos")
        videos = cursor.fetchall()

        # Populate the ComboBox with video descriptions
        self.video_combo.clear()
        for video in videos:
            self.video_combo.addItem(video[1], video[2])  # Display description, store video path

        conn.close()

    # Function to load images from the database into the ComboBox
    def load_images_from_database(self):
        conn = sqlite3.connect('first_fire.db')  # Connect to your actual database
        cursor = conn.cursor()

        # Fetch image records
        cursor.execute("SELECT id, description, image_path FROM images")
        images = cursor.fetchall()

        # Populate the ComboBox with image descriptions
        self.image_combo.clear()
        for image in images:
            self.image_combo.addItem(image[1], image[2])  # Display description, store image path

        conn.close()

    # Function to analyze the selected video for fire
    def analyze_video(self):
        video_path = self.video_combo.currentData()  # Get the selected video path

        if video_path:
            self.result_label.setText("Analyzing video for fire detection...")
            start_time = time.time()

            self.process_video(video_path)
            end_time = time.time()
            runtime = end_time - start_time
            print(f"Video analysis runtime: {runtime:.2f} seconds")
        else:
            QMessageBox.warning(self, "Error", "No video selected.")

    # Function to process video frames in parallel and display fire-detected frame
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fire_detected = False
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Stop when the video ends

            # Convert the OpenCV BGR frame to RGB (Pillow works with RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to Pillow Image
            pillow_image = Image.fromarray(frame_rgb)
            # Display the frame (optional)
            #pillow_image.show()
            new_size = (640, 640)
            pillow_image_resized = pillow_image.resize(new_size)
            # Resize the frame to the model's input size if needed
            #frame_resized = cv2.resize(frame, (640, 640))

            # Inference on the frame
            results = model(pillow_image_resized)

            # Check if fire is detected in the frame
            detections = results.xyxy[0].cpu().numpy()  # Get detection results

            # Convert the Pillow image to a NumPy array (which OpenCV understands)
            frame_resized = np.array(pillow_image_resized)
            # Since Pillow and OpenCV use different color formats, convert from RGB to BGR
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)

            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if int(detection[5]) == 0:  # Assuming class '0' is fire
                    fire_detected = True
                    # Draw bounding box on the image
                    cv2.rectangle(frame_resized, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(frame_resized, f'Fire {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            frame_count += 1

            # Break early if fire is detected
            if fire_detected:
                break

        cap.release()

        if fire_detected:
            self.result_label.setText(f"Fire detected in video '{self.video_combo.currentText()}'!")
        else:
            self.result_label.setText(f"No fire detected in video '{self.video_combo.currentText()}'.")

        self.display_frame(frame_resized)

    # Function to process a single frame
    def process_frame(self, frame):
        # Resize the frame to the model's input size if needed
        frame_resized = cv2.resize(frame, (640, 640))

        # Inference on the frame
        results = model(frame_resized)

        # Check if fire is detected in the frame
        detections = results.xyxy[0].cpu().numpy()  # Get detection results

        # Draw bounding boxes for detected fires
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == 0:  # Assuming class '0' is fire
                # Draw bounding box on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f'Fire {conf:.2f}', (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                return frame, True  # Fire detected

        return frame, False  # No fire detected

    # Function to analyze the selected image for fire
    def analyze_image(self):
        image_path = self.image_combo.currentData()  # Get the selected image path

        if image_path:
            self.result_label.setText("Analyzing image for fire detection...")

            # Load and analyze the image
            # img = cv2.imread(image_path)
            # img_resized = cv2.resize(img, (640, 640))
            with Image.open(image_path) as img:

                new_size = (640, 640)
                # Resize the image
                img_resized = img.resize(new_size)

                # Save the resized image
                #resized_img.save('resized_image.jpg')

                # Show the resized image (optional)
                #img_resized.show()
            # Inference on the image
            # Start measuring time
            start_time = time.time()
            # End measuring time

            results = model(img_resized)
            end_time = time.time()
            runtime = end_time - start_time
            # Calculate and display the runtime
            print(f"Image analysis runtime: {runtime:.2f} seconds")
            #results.show()
            # Check if fire is detected in the image and draw bounding boxes
            detections = results.xyxy[0].cpu().numpy()
            fire_detected = False
            # Convert the Pillow image to a NumPy array (which OpenCV understands)
            resized_img_cv2 = np.array(img_resized)
            # Since Pillow and OpenCV use different color formats, convert from RGB to BGR
            resized_img_cv2 = cv2.cvtColor(resized_img_cv2, cv2.COLOR_RGB2BGR)
            # Loop through detections
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                if int(cls) == 0:  # Assuming class '0' is fire
                    fire_detected = True
                    # Draw bounding box on the image
                    cv2.rectangle(resized_img_cv2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                    cv2.putText(resized_img_cv2, f'Fire {conf:.2f}', (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # If fire is detected, display the image with bounding boxes
            if fire_detected:
                self.result_label.setText(f"Fire detected in image '{self.image_combo.currentText()}'!")
            else:
                self.result_label.setText(f"No fire detected in image '{self.image_combo.currentText()}'.")

            # Show the image with bounding boxes (even if no fire is detected)
            self.display_frame(resized_img_cv2)

        else:
            QMessageBox.warning(self, "Error", "No image selected.")

    # Function to display a frame in the QLabel
    def display_frame(self, frame):
        # Convert OpenCV BGR format to RGB format
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set the QImage to QLabel
        pixmap = QPixmap.fromImage(q_img)
        self.frame_label.setPixmap(pixmap.scaled(self.frame_label.size(), Qt.KeepAspectRatio))


class RunSimulationWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Fire Spread Simulation')
        self.setGeometry(100, 100, 800, 600)

        # Layout
        layout = QVBoxLayout(self)

        # QLabel for status
        self.label = QLabel('Fire Spread Simulation Running...', self)
        layout.addWidget(self.label)

        # Input field for simulation time
        self.time_input = QLineEdit(self)
        self.time_input.setPlaceholderText("Enter simulation time in minutes")
        layout.addWidget(self.time_input)

        # ComboBox to select fire-detected frame from the database
        self.frame_combo = QComboBox(self)
        self.load_frames_from_database()
        layout.addWidget(self.frame_combo)

        # Button to start the simulation
        self.start_button = QPushButton('Start Simulation', self)
        self.start_button.clicked.connect(self.start_simulation)
        layout.addWidget(self.start_button)

        # Fire grid initialization
        self.rows, self.cols = 100, 100
        self.fire_grid = np.zeros((self.rows, self.cols), dtype=int)

        # Matplotlib canvas for grid visualization
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)
        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)


        # Fire spread parameters
        # self.wind_direction = 270  # wind direction in degrees (0: east, 90: north)
        # self.wind_speed = 1.5  # higher speed increases spread probability
        # self.temperature = 30  # degrees Celsius
        # self.humidity = 20  # percentage
        # self.slope_factor = 1.2  # factor to increase spread uphill

        # Vegetation map (simplified, 0: low, 1: medium, 2: high fuel load)
        self.vegetation = np.random.randint(0, 2, size=(self.rows, self.cols))

        print(f"Running simulation with parameters:\n"
              f"Wind Direction: {wind_direction} degrees\n"
              f"Wind Speed: {wind_speed} m/s\n"
              f"Temperature: {temperature} °C\n"
              f"Humidity: {humidity}%\n"
              f"Slope Factor: {slope_factor}\n"
              f"Vegetation: {self.vegetation}")

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    def load_frames_from_database(self):
        conn = sqlite3.connect('first_fire.db')  # Connect to your actual database
        cursor = conn.cursor()

        # Fetch frame records
        cursor.execute("SELECT id, description, image_path FROM images")
        frames = cursor.fetchall()

        self.frame_combo.clear()
        for frame in frames:
            self.frame_combo.addItem(frame[1], frame[2])  # Display description, store image path

        conn.close()

    def start_simulation(self):
        if not self.time_input.text().isdigit():
            QMessageBox.warning(self, "Input Error", "Please enter a valid number for simulation time.")
            return

        simulation_time = int(self.time_input.text())  # Get the simulation time in minutes

        if not self.frame_combo.currentData():
            QMessageBox.warning(self, "Selection Error", "No frame selected.")
            return

        image_path = self.frame_combo.currentData()  # Get the selected frame path
        self.fire_grid = self.load_frame(image_path)

        # Calculate the number of steps based on simulation time
        steps = self.calculate_simulation_steps(simulation_time)

        # Run the simulation directly for the number of steps
        for _ in range(steps):
            self.fire_grid = self.update_grid(self.fire_grid)

        # Visualize the final grid state after the given time
        self.visualize_grid()

    def calculate_simulation_steps(self, simulation_time):
        """Calculate the number of steps based on the given simulation time in minutes."""
        grass_fires = 14400  # Grass fires can spread very quickly, often up to 14.4 km/h
        forest_fires = 1200  # Forest fires spread more slowly, from 0.8 to 1.6 km/h
        brush_shrub_fires = 7200  # Intermediate rates around 4.8 to 9.6 km/h

        spread_rate = forest_fires  # Example: use forest fire spread rate

        # Adjust for Grid Cell Size (e.g., each grid cell is 10x10 meters instead of 1x1 meter)
        cell_size = 1  # Cell size in meters
        spread_rate_per_minute = (spread_rate * 1000) / 60  # Convert spread rate to meters per minute
        spread_rate_per_minute /= cell_size  # Adjust for cell size

        # Calculate the number of steps for the simulation time
        simulation_steps = int(simulation_time * spread_rate_per_minute)

        return simulation_steps

    def load_frame(self, image_path):
        """Load and initialize the fire grid based on YOLOv5 fire detection bounding boxes."""
        # Load the YOLOv5 model
        #model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt', source='local')

        # Read and process the image
        img = cv2.imread(image_path)
        img_height, img_width, _ = img.shape

        # Inference
        results = model(image_path)
        results.show()

        # Extract bounding boxes for detected fires
        detections = results.xyxy[0].cpu().numpy()

        # Initialize fire grid
        fire_grid = np.zeros((self.rows, self.cols), dtype=int)

        # Map detections to grid cells
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            if cls == 0:  # Assuming '0' is the class for fire
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Center of the bounding box
                gx, gy = cx * self.cols // img_width, cy * self.rows // img_height
                fire_grid[gy, gx] = 1

        return fire_grid

    def update_grid(self, grid):
        """Update the fire grid based on spread rules with realistic spread adjustments."""
        new_grid = grid.copy()
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        direction_angles = [90, 45, 0, 315, 270, 225, 180, 135]

        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r, c] == 1:  # burning
                    new_grid[r, c] = 2  # burned
                    for (dr, dc), spread_angle in zip(directions, direction_angles):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.rows and 0 <= nc < self.cols and grid[nr, nc] == 0:
                            # Adjust spread probability based on factors
                            prob = 0.1  # Reduced base probability to slow down spread
                            angle_influence = np.cos(np.radians(self.angle_diff(spread_angle, wind_direction)))
                            prob *= (1 + wind_speed * angle_influence)  # wind effect
                            prob *= (1 + (temperature - 20) / 30)  # temperature effect
                            prob *= (1 - humidity / 100)  # humidity effect
                            prob *= slope_factor if dr == -1 else 1  # uphill spread
                            prob *= 1 + self.vegetation[nr, nc] * 0.5  # vegetation effect
                            if np.random.rand() < prob:
                                new_grid[nr, nc] = 1
        return new_grid

    def angle_diff(self, angle1, angle2):
        """Calculate the smallest difference between two angles in degrees."""
        return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

    def visualize_grid(self):
        """Visualize the fire spread grid."""
        self.ax.clear()
        self.ax.imshow(self.fire_grid, cmap='hot', interpolation='nearest')
        self.ax.set_title('Fire Spread Simulation')
        self.canvas.draw()

class HelpWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("User Help")
        self.setGeometry(200, 200, 500, 400)

        layout = QVBoxLayout()

        # Help text
        help_text = QLabel("""
        Welcome to the Fire Detection and Simulation Application.

        1. **Select Database**: Choose or create a database to store data.
        2. **Upload Data**: Upload images or videos for fire detection.
        3. **Process Data**: Analyze uploaded media for fire detection using a trained model.
        4. **Set Environmental Parameters**: Configure settings like wind speed, temperature, and humidity.
        5. **Run Fire Spread Simulation**: Simulate how fire will spread over time based on the selected frame and environmental factors.

        For further assistance, please contact support.
        """, self)
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        self.setLayout(layout)
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(300, 300, 900, 1200)

        layout = QVBoxLayout()

        button1 = QPushButton("Select Database")
        button2 = QPushButton("Edit Video Details")
        button3 = QPushButton("Upload Data")
        button4 = QPushButton("Process Data")
        button5 = QPushButton("Select Environment Parameters")
        button6 = QPushButton("Run Simulation")



        button1.clicked.connect(lambda _, : self.open_select_database_window())
        button2.clicked.connect(lambda _, : self.open_edit_video_window())
        button3.clicked.connect(lambda _, : self.open_upload_data_window())
        button4.clicked.connect(lambda _, : self.open_process_video_window())

        button5.clicked.connect(lambda _, : self.open_select_environment_parameters_window())
        button6.clicked.connect(lambda _, : self.open_run_simulation_window())


        layout.addWidget(button1)
        layout.addWidget(button2)
        layout.addWidget(button3)
        layout.addWidget(button4)

        layout.addWidget(button5)
        layout.addWidget(button6)

        # Create a horizontal layout for the Help button and spacer
        help_layout = QHBoxLayout()
        help_button = QPushButton("?")  # Small Help button
        help_button.setFixedSize(30, 30)  # Set the size to be smaller (width, height)
        help_button.clicked.connect(self.open_help_window)
        # Spacer to push the Help button to the right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        help_layout.addItem(spacer)
        help_layout.addWidget(help_button)

        # Add the help layout to the main layout
        layout.addLayout(help_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.help_window.show()
    def open_select_database_window(self):
        self.database_window = SelectDatabaseWindow()
        self.database_window.show()
    def open_edit_video_window(self):
        self.database_window = EditVideoWindow()
        self.database_window.show()
    def open_upload_data_window(self):
        self.database_window = UploadDataWindow()
        self.database_window.show()
    def open_process_video_window(self):
        self.database_window = ProcessVideoWindow()
        self.database_window.show()
    def open_select_environment_parameters_window(self):
        self.database_window = SelectEnvironmentParametersWindow()
        self.database_window.show()
    def open_run_simulation_window(self):
        self.database_window = RunSimulationWindow()
        self.database_window.show()

    def open_window(self, window_number):
        self.secondary_window = SecondaryWindow(window_number)
        self.secondary_window.show()



app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())




#matplotlib.use('Qt5Agg')  # or 'Agg', 'Qt5Agg', etc. depending on your environment


model = torch.hub.load('./yolov5', 'custom', path='./yolov5/runs/train/exp5/weights/best.pt', source='local')
# Image
img_path = './yolov5/images.jpeg'
img = cv2.imread(img_path)
img_height, img_width, _ = img.shape
# Inference
results = model(img_path)
# Results, change the flowing to: results.show()
results.show()  # or .show(), .save(), .crop(), .pandas(), etc


# Extract bounding box coordinates of detected fires
detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class

#print(detections)



# Initialize fire spread grid
rows, cols = 100, 100

fire_grid = np.zeros((rows, cols), dtype=int)

# Map detections to grid cells
for detection in detections:
    x1, y1, x2, y2, conf, cls = detection
    if cls == 0:  # Assuming '0' is the class for fire
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2) # center of the box
        gx, gy = cx * cols // img_width, cy * rows // img_height
        fire_grid[gy, gx] = 1

print(fire_grid)# Grid dimensions

# Parameters (simplified)
wind_direction = 270  # wind direction in degrees (0: east, 90: north)
wind_speed = 1.5  # higher speed increases spread probability
temperature = 30  # degrees Celsius
humidity = 20  # percentage
slope_factor = 1.2  # factor to increase spread uphill

# Vegetation map (simplified, 0: low, 1: medium, 2: high fuel load)
vegetation = np.random.randint(0, 2, size=(rows, cols))

# Direction vectors for neighbors (N, NE, E, SE, S, SW, W, NW)
directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
direction_angles = [90, 45, 0, 315, 270, 225, 180, 135]

def angle_diff(angle1, angle2):
    """Calculate the smallest difference between two angles in degrees."""
    return min(abs(angle1 - angle2), 360 - abs(angle1 - angle2))

def update_grid(grid):
    new_grid = grid.copy()
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 1:  # burning
                new_grid[r, c] = 2  # burned
                # Spread fire to neighbors
                for (dr, dc), spread_angle in zip(directions, direction_angles):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
                        # Adjust spread probability based on factors
                        prob = 0.3  # base probability
                        angle_influence = np.cos(np.radians(angle_diff(spread_angle, wind_direction)))
                        prob *= (1 + wind_speed * angle_influence)  # wind effect
                        prob *= (1 + (temperature - 20) / 30)  # temperature effect
                        prob *= (1 - humidity / 100)  # humidity effect
                        prob *= slope_factor if dr == -1 else 1  # uphill spread
                        prob *= 1 + vegetation[nr, nc] * 0.5  # vegetation effect
                        if np.random.rand() < prob:
                            new_grid[nr, nc] = 1
    return new_grid


def visualize_grid(grid, save_path=None):
    #ax1 = plt.imshow(grid)


    plt.imshow(grid, cmap='hot', interpolation='nearest')
    plt.title('Fire Spread Simulation')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()


# Run and visualize the simulation

grass_fires = 14400 # Grass fires can spread very quickly often at rates of up to 14.4 kilometers per hour
forest_fires = 1200 # The spread rate in forested areas is usually slower than in grasslands, typically ranging from 0.8 to 1.6 kilometers per hour
brush_shrub_fires = 7200 # These can spread at intermediate rates, often around 4.8 to 9.6 kilometers per hour

spread_rate = forest_fires

# Convert Spread Rate to Meters per Minute
spread_rate_MpM = spread_rate/60

# Define Grid Cell Size: Assume each grid cell represents a 1x1 meter area.
# Calculate Time per Step
simulation_step = 1/spread_rate_MpM

real_time = 2 # fire spread after 3 minutes
steps = real_time * spread_rate_MpM
#print(steps)

for step in range(int(steps)):  # number of time steps
    fire_grid = update_grid(fire_grid)
    # visualize_grid(grid)

# Save the final grid as an image
visualize_grid(fire_grid, save_path='fire_spread_simulation.png')


