import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                              QHBoxLayout, QLabel, QLineEdit, QPushButton,
                              QSlider, QComboBox, QProgressBar, QTextEdit,
                              QFileDialog, QSpinBox, QFormLayout)
from PySide6.QtCore import Qt, QThreadPool, QRunnable, QObject, Signal
from PySide6.QtGui import QDoubleValidator
from deepface import DeepFace
from retinaface import RetinaFace
import faiss
import torch
from PIL import Image, ImageFile
import glob
import numpy as np
import shutil
import pandas as pd
import pickle

ImageFile.LOAD_TRUNCATED_IMAGES = True

class WorkerSignals(QObject):
    progress = Signal(int)
    message = Signal(str)
    finished = Signal(pd.DataFrame)

class FaceProcessor(QRunnable):
    def __init__(self, input_path, image_dir, output_dir, threshold, device, batch_size):
        super().__init__()
        self.input_path = input_path
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.threshold = threshold
        self.device = device
        self.batch_size = self.calculate_batch_size(batch_size)
        self.signals = WorkerSignals()
        self.backends = {
            "face_detector": "retinaface", 
            "face_recognizer": "ArcFace"
        }

    def calculate_batch_size(self, user_set_size):
        if self.device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            used_mem = torch.cuda.memory_allocated(0)
            free_mem = total_mem - used_mem
            # Estimate 500MB per batch item (adjust based on actual model memory usage)
            safe_batch = int(free_mem // (500 * 1024**2))
            return min(user_set_size, safe_batch) if user_set_size > 0 else safe_batch
        return user_set_size if user_set_size > 0 else 4

    def run(self):
        try:
            # Use DeepFace's built-in functions for model handling
            self.signals.message.emit(f"Initializing with {self.backends['face_detector']} detector and {self.backends['face_recognizer']} recognition model...")
            
            # Set device if CUDA is selected
            if self.device == "cuda" and torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                self.signals.message.emit("Using CUDA for processing")
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                self.signals.message.emit("Using CPU for processing")

            # Collect all images in the directory
            extensions = ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp']
            image_files = []
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(self.image_dir, '**/*.' + ext), recursive=True))

            if not image_files:
                self.signals.message.emit("No images found in the directory.")
                return

            # Create output directory if needed
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Build FAISS index
            index = faiss.IndexFlatL2(512)
            image_paths = []
            total = len(image_files)
            
            self.signals.message.emit(f"Processing {total} images...")
            
            for i in range(0, total, self.batch_size):
                batch = image_files[i:i + self.batch_size]
                embeddings = []
                batch_paths = []
                for img_path in batch:
                    try:
                        # FIX: Use RetinaFace directly as a module function
                        faces = RetinaFace.extract_faces(img_path=img_path, align=True)
                        if not faces or len(faces) == 0:
                            self.signals.message.emit(f"No faces found in {img_path}.")
                            continue
                        # Convert to tensor and move to device
                        face = Image.fromarray(faces[0]).convert('RGB')
                        face = face.resize((112, 112))
                        face = np.array(face)
                        
                        # Get embedding using DeepFace
                        embedding = DeepFace.represent(face, model_name="ArcFace", enforce_detection=False)
                        
                        if isinstance(embedding, list):
                            embedding = np.array(embedding[0]["embedding"])
                        
                        embeddings.append(embedding)
                        batch_paths.append(img_path)
                    except Exception as e:
                        self.signals.message.emit(f"Error processing {img_path}: {str(e)}")
                
                if embeddings:
                    embeddings_array = np.array(embeddings).astype('float32')
                    index.add(embeddings_array)
                    image_paths.extend(batch_paths)
                
                self.signals.progress.emit(int((i + len(batch)) / total * 100))

            # Process input image
            try:
                # FIX: Use proper RetinaFace method for detection
                input_faces = RetinaFace.extract_faces(self.input_path, align=True)
                if not input_faces or len(input_faces) == 0:
                    self.signals.message.emit("No faces found in input image.")
                    return
            except Exception as e:
                self.signals.message.emit(f"Error extracting faces from input image: {str(e)}")
                return

            # Prepare results
            results = []
            for face_idx, face in enumerate(input_faces):
                try:
                    face_img = Image.fromarray(face).convert('RGB')
                    face_img = face_img.resize((112, 112))
                    face_array = np.array(face_img)
                    
                    # Get embedding using DeepFace
                    query = DeepFace.represent(face_array, model_name="ArcFace", enforce_detection=False)
                    
                    if isinstance(query, list):
                        query = np.array(query[0]["embedding"])
                    
                    query = np.array([query]).astype('float32')
                    
                    if len(image_paths) > 0:
                        distances, indices = index.search(query, k=min(5, len(image_paths)))
                        
                        for i in range(len(indices[0])):
                            idx = indices[0][i]
                            distance = distances[0][i]
                            similarity_score = 1 - distance
                            
                            if similarity_score > self.threshold:
                                src = image_paths[idx]
                                base_name = os.path.basename(src)
                                dst = os.path.join(self.output_dir, f"face{face_idx+1}_match{i+1}_{base_name}")
                                shutil.copy2(src, dst)
                                results.append({'path': src, 'score': similarity_score})
                except Exception as e:
                    self.signals.message.emit(f"Error processing input face {face_idx+1}: {str(e)}")

            # Convert to DataFrame and emit
            df = pd.DataFrame(results)
            if len(df) > 0:
                self.signals.finished.emit(df)
            else:
                self.signals.message.emit("No similar faces found above the threshold.")
                empty_df = pd.DataFrame(columns=['path', 'score'])
                self.signals.finished.emit(empty_df)
                
        except Exception as e:
            self.signals.message.emit(f"Error: {str(e)}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.threadpool = QThreadPool()
        self.current_matrix = pd.DataFrame()

    def init_ui(self):
        self.setWindowTitle("Face Similarity Finder")
        self.setGeometry(100, 100, 800, 600)

        self.create_input_widgets()
        self.create_controls()
        self.create_progress()
        self.create_log()

        layout = QVBoxLayout()
        layout.addLayout(self.input_layout)
        layout.addLayout(self.control_layout)
        layout.addLayout(self.progress_layout)
        layout.addWidget(self.log_output)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_input_widgets(self):
        # Input image selection
        input_image_layout = QHBoxLayout()
        self.input_image_line = QLineEdit()
        self.input_image_line.setPlaceholderText("Select input image")
        input_image_button = QPushButton("Browse...")
        input_image_button.clicked.connect(self.browse_input_image)
        input_image_layout.addWidget(self.input_image_line)
        input_image_layout.addWidget(input_image_button)

        # Image directory selection
        image_dir_layout = QHBoxLayout()
        self.image_dir_line = QLineEdit()
        self.image_dir_line.setPlaceholderText("Select image directory")
        image_dir_button = QPushButton("Browse...")
        image_dir_button.clicked.connect(self.browse_image_dir)
        image_dir_layout.addWidget(self.image_dir_line)
        image_dir_layout.addWidget(image_dir_button)

        # Output directory selection
        output_dir_layout = QHBoxLayout()
        self.output_dir_line = QLineEdit()
        self.output_dir_line.setPlaceholderText("Select output directory")
        output_dir_button = QPushButton("Browse...")
        output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_line)
        output_dir_layout.addWidget(output_dir_button)

        # Threshold controls
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold (0-1):")
        self.threshold_line = QLineEdit("0.5")
        self.threshold_line.setFixedWidth(50)
        self.threshold_line.setAlignment(Qt.AlignRight)
        # FIX: Use proper validator with locale
        validator = QDoubleValidator(0.0, 1.0, 2)
        validator.setNotation(QDoubleValidator.StandardNotation)
        self.threshold_line.setValidator(validator)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(50)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_line)
        threshold_layout.addWidget(self.threshold_slider)
        
        # FIX: Connect signals after UI is created
        self.threshold_slider.valueChanged.connect(self.update_threshold_line)
        self.threshold_line.editingFinished.connect(self.update_threshold_slider)

        # Device selection
        device_layout = QHBoxLayout()
        device_label = QLabel("Device:")
        self.device_combo = QComboBox()
        if torch.cuda.is_available():
            self.device_combo.addItems(["CPU", "CUDA"])
        else:
            self.device_combo.addItems(["CPU"])
        self.device_combo.currentIndexChanged.connect(self.update_batch_size)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)

        # Batch size
        batch_size_layout = QHBoxLayout()
        batch_size_label = QLabel("Batch Size:")
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 100)
        self.batch_size_spin.setValue(16)
        batch_size_layout.addWidget(batch_size_label)
        batch_size_layout.addWidget(self.batch_size_spin)

        # Create form layout
        self.input_layout = QVBoxLayout()
        
        # Add all rows
        self.input_layout.addWidget(QLabel("Input Image:"))
        self.input_layout.addLayout(input_image_layout)
        
        self.input_layout.addWidget(QLabel("Image Directory:"))
        self.input_layout.addLayout(image_dir_layout)
        
        self.input_layout.addWidget(QLabel("Output Directory:"))
        self.input_layout.addLayout(output_dir_layout)
        
        self.input_layout.addLayout(threshold_layout)
        self.input_layout.addLayout(device_layout)
        self.input_layout.addLayout(batch_size_layout)
        
        # Update batch size after UI is created
        self.update_batch_size()

    def create_controls(self):
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        self.control_layout = control_layout

    def create_progress(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        progress_layout = QVBoxLayout()
        progress_layout.addWidget(QLabel("Indexing Progress:"))
        progress_layout.addWidget(self.progress_bar)
        self.progress_layout = progress_layout

    def create_log(self):
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

    def browse_input_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "", 
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.webp *.tiff)"
        )
        if file_path:
            self.input_image_line.setText(file_path)

    def browse_image_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.image_dir_line.setText(dir_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_line.setText(dir_path)

    def update_threshold_line(self, value):
        threshold = value / 100.0
        # FIX: Block signals to prevent recursion
        self.threshold_line.blockSignals(True)
        self.threshold_line.setText(f"{threshold:.2f}")
        self.threshold_line.blockSignals(False)

    def update_threshold_slider(self):
        try:
            text = self.threshold_line.text()
            threshold = float(text)
            if 0.0 <= threshold <= 1.0:
                # FIX: Block signals to prevent recursion
                self.threshold_slider.blockSignals(True)
                self.threshold_slider.setValue(int(threshold * 100))
                self.threshold_slider.blockSignals(False)
        except ValueError:
            # Reset to previous valid value if conversion fails
            self.update_threshold_line(self.threshold_slider.value())

    def update_batch_size(self, index=None):
        device = self.device_combo.currentText().lower()
        if device == "cuda" and torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(0).total_memory
            free_mem = total_mem - torch.cuda.memory_allocated(0)
            # Estimate memory per batch item more conservatively
            max_batch = max(1, int(free_mem // (1000 * 1024**2)))
            self.batch_size_spin.setRange(1, min(max_batch, 100))
            self.batch_size_spin.setValue(min(16, max_batch))
        else:
            self.batch_size_spin.setRange(1, 16)
            self.batch_size_spin.setValue(4)

    def start_processing(self):
        input_path = self.input_image_line.text()
        image_dir = self.image_dir_line.text()
        output_dir = self.output_dir_line.text()
        
        # Validation
        if not input_path:
            self.log_output.append("Error: Input image not selected.")
            return
        if not image_dir:
            self.log_output.append("Error: Image directory not selected.")
            return
        if not output_dir:
            self.log_output.append("Error: Output directory not selected.")
            return
            
        if not os.path.isfile(input_path):
            self.log_output.append("Error: Input image not found.")
            return
        if not os.path.isdir(image_dir):
            self.log_output.append("Error: Image directory not found.")
            return
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
                self.log_output.append(f"Created output directory: {output_dir}")
            except Exception as e:
                self.log_output.append(f"Error creating output directory: {str(e)}")
                return

        # Get parameters
        threshold = float(self.threshold_line.text())
        device = self.device_combo.currentText().lower()
        batch_size = self.batch_size_spin.value()

        # Disable UI elements during processing
        self.start_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.append("Starting face similarity processing...")

        # Create and start the worker
        worker = FaceProcessor(input_path, image_dir, output_dir, threshold, device, batch_size)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.message.connect(self.log_message)
        worker.signals.finished.connect(self.process_results)
        self.threadpool.start(worker)

    def log_message(self, message):
        self.log_output.append(message)
        # Scroll to bottom to show latest message
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def process_results(self, df):
        self.current_matrix = df
        
        if len(df) > 0:
            self.log_output.append(f"Complete! Found {len(df)} similar faces.")
            # Display top matches
            sorted_df = df.sort_values(by=['score'], ascending=False)
            if len(sorted_df) > 0:
                self.log_output.append("\nTop matches:")
                for i, (_, row) in enumerate(sorted_df.iterrows()):
                    if i < 5:  # Show top 5 matches
                        file_name = os.path.basename(row['path'])
                        self.log_output.append(f"  {file_name} - Score: {row['score']:.4f}")
            
            # Save to .pkl
            try:
                similarity_file = os.path.join(self.output_dir_line.text(), "similarity_matrix.pkl")
                with open(similarity_file, "wb") as f:
                    pickle.dump(df, f)
                self.log_output.append(f"\nSimilarity matrix saved to {similarity_file}")
            except Exception as e:
                self.log_output.append(f"Error saving similarity matrix: {str(e)}")
        else:
            self.log_output.append("No similar faces found above the threshold.")
        
        self.progress_bar.setValue(100)
        self.start_button.setEnabled(True)
        self.log_output.append("\nProcessing complete! Results saved to output directory.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())