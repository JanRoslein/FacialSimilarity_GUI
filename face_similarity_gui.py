import os
import sys
import shutil
import pickle
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import tensorflow as tf
from deepface import DeepFace
from retinaface import RetinaFace
from tqdm import tqdm
import warnings

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QGridLayout, QLabel, QLineEdit, 
                             QPushButton, QFileDialog, QMessageBox, QProgressBar,
                             QComboBox, QSpinBox, QCheckBox, QGroupBox, QFrame,
                             QScrollArea, QTextEdit, QSplitter, QTabWidget)
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QPixmap, QFont, QPalette, QColor, QIcon
from PySide6.QtWidgets import QGraphicsDropShadowEffect

# Suppress warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setup_style()
        
    def setup_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #667eea, stop:1 #764ba2);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 12px 24px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5a6fd8, stop:1 #6a4190);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4c5bc4, stop:1 #5d3a7e);
                }
                QPushButton:disabled {
                    background: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #f8f9fa;
                    color: #495057;
                    border: 2px solid #e9ecef;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: 500;
                    font-size: 13px;
                }
                QPushButton:hover {
                    background: #e9ecef;
                    border-color: #adb5bd;
                }
                QPushButton:pressed {
                    background: #dee2e6;
                }
            """)

class ModernLineEdit(QLineEdit):
    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                background: white;
                color: #343a40;
            }
            QLineEdit:focus {
                border-color: #667eea;
                outline: none;
            }
        """)

class ModernComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 12px 16px;
                font-size: 14px;
                background: white;
                color: #343a40;
                min-width: 150px;
            }
            QComboBox:focus {
                border-color: #667eea;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOCIgdmlld0JveD0iMCAwIDEyIDgiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDFMNiA2TDExIDEiIHN0cm9rZT0iIzY2N2VlYSIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz4KPC9zdmc+);
            }
        """)

class ModernGroupBox(QGroupBox):
    def __init__(self, title):
        super().__init__(title)
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 16px;
                color: #343a40;
                border: 2px solid #e9ecef;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px 0 8px;
                background: white;
            }
        """)

class ProcessingThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    finished_processing = Signal(list, pd.DataFrame)
    error_occurred = Signal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
    
    def create_pillow_database_representations(self, db_path):
        """Create database representations using Pillow to handle Unicode filenames"""
        representations = []
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
        
        try:
            for root, dirs, files in os.walk(db_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            # Load image using Pillow
                            pil_image = Image.open(file_path)
                            if pil_image.mode != 'RGB':
                                pil_image = pil_image.convert('RGB')
                            
                            representations.append({
                                'path': file_path,
                                'image': np.array(pil_image)
                            })
                        except Exception as e:
                            print(f"Failed to load {file_path}: {e}")
                            continue
            
            return representations
        except Exception as e:
            self.error_occurred.emit(f"Error creating database representations: {e}")
            return []

    def process_with_individual_verification(self, target_image, db_path, threshold, output_folder):
        """Alternative processing method using individual verification"""
        try:
            results = []
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
            
            # Get all image files in database
            db_images = []
            for root, dirs, files in os.walk(db_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        db_images.append(os.path.join(root, file))
            
            if not db_images:
                return [], pd.DataFrame()
            
            # Process each image individually
            verification_results = []
            for db_image_path in db_images:
                try:
                    # Load database image using Pillow
                    db_pil_image = Image.open(db_image_path)
                    if db_pil_image.mode != 'RGB':
                        db_pil_image = db_pil_image.convert('RGB')
                    db_image_array = np.array(db_pil_image)
                    
                    # Verify similarity
                    result = DeepFace.verify(
                        img1_path=target_image,
                        img2_path=db_image_array,
                        model_name="ArcFace",
                        distance_metric="cosine",
                        detector_backend="retinaface",
                        enforce_detection=False,
                        align=True
                    )
                    
                    similarity_score = 1 - result['distance']
                    if similarity_score >= threshold:
                        original_filename = os.path.basename(db_image_path)
                        output_path = os.path.join(output_folder, original_filename)
                        
                        # Copy file using Pillow to handle Unicode
                        if not os.path.exists(output_path):
                            db_pil_image.save(output_path)
                        
                        results.append({
                            'image_path': db_image_path,
                            'score': similarity_score
                        })
                        
                        verification_results.append({
                            'identity': db_image_path,
                            'distance': result['distance'],
                            'threshold': result['threshold']
                        })
                        
                except Exception as e:
                    print(f"Failed to process {db_image_path}: {e}")
                    continue
            
            # Create DataFrame from verification results
            if verification_results:
                df = pd.DataFrame(verification_results)
            else:
                df = pd.DataFrame()
            
            # Save CSV results
            if results:
                csv_path = os.path.join(output_folder, 'similarity_results.csv')
                results_df = pd.DataFrame(results)
                results_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            return results, df
            
        except Exception as e:
            self.error_occurred.emit(f"Error in individual verification: {str(e)}")
            return [], pd.DataFrame()

    def load_image_with_pillow(self, image_path):
        """Load image using Pillow with proper error handling for Unicode paths"""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            return None


    def run(self):
        try:
            self.status_updated.emit("Initializing GPU detection...")
            
            # GPU Detection
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus and self.params['use_gpu']:
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    self.status_updated.emit(f"Using GPU: {gpus[0].name}")
                except RuntimeError as e:
                    self.status_updated.emit(f"GPU setup failed, using CPU: {e}")
            else:
                tf.config.set_visible_devices([], 'GPU')
                self.status_updated.emit("Using CPU for processing")

            self.status_updated.emit("Loading and aligning faces...")
            aligned_faces = self.align_faces(self.params['image_path'])
            
            if not aligned_faces:
                self.error_occurred.emit("Failed to process input image or no faces detected")
                return

            results = []
            total_faces = len(aligned_faces)
            
            for i, aligned_image in enumerate(aligned_faces):
                self.status_updated.emit(f"Processing face {i+1}/{total_faces}")
                
                result_set, dfs = self.process_image_traditional(
                    aligned_image, 
                    self.params['db_path'], 
                    self.params['threshold'], 
                    self.params['output_folder']
                )
                
                results.append((result_set, dfs))
                progress = int(((i + 1) / total_faces) * 100)
                self.progress_updated.emit(progress)

            # Combine results
            all_results = [result for result_set, _ in results for result in result_set]
            valid_dfs = [dfs for _, dfs in results if not dfs.empty]
            
            if valid_dfs:
                combined_dfs = pd.concat(valid_dfs, ignore_index=True)
            else:
                combined_dfs = pd.DataFrame()
            
            self.finished_processing.emit(all_results, combined_dfs)
            
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")

    def align_faces(self, image_path):
        try:
            # Use Pillow settings for large images
            if self.params['allow_large_images']:
                Image.MAX_IMAGE_PIXELS = None
            
            # Load image using Pillow to handle Unicode paths
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert PIL image to numpy array for RetinaFace
            image_array = np.array(pil_image)
            
            # Use RetinaFace with numpy array instead of file path
            faces = RetinaFace.extract_faces(img_path=image_array, align=True)
            
            # Convert faces to PIL Images
            aligned_faces = []
            for face in faces:
                # RetinaFace returns values in [0,1] range
                if face.max() <= 1.0:
                    face_uint8 = (face * 255).astype(np.uint8)
                else:
                    face_uint8 = face.astype(np.uint8)
                aligned_faces.append(Image.fromarray(face_uint8))
            
            return aligned_faces
        except Exception as e:
            self.error_occurred.emit(f"Error processing image {image_path}: {e}")
            return []

    def process_image_traditional(self, aligned_image, db_path, threshold, output_folder):
        try:
            # Convert PIL image to numpy array
            aligned_image_np = np.array(aligned_image)
            
            # Ensure the image is in RGB format and correct data type
            if len(aligned_image_np.shape) == 3 and aligned_image_np.shape[2] == 3:
                # Image is already RGB
                pass
            elif len(aligned_image_np.shape) == 2:
                # Convert grayscale to RGB
                aligned_image_np = np.stack([aligned_image_np] * 3, axis=-1)
            
            # Ensure correct data type (uint8, 0-255 range)
            if aligned_image_np.dtype != np.uint8:
                if aligned_image_np.max() <= 1.0:
                    aligned_image_np = (aligned_image_np * 255).astype(np.uint8)
                else:
                    aligned_image_np = aligned_image_np.astype(np.uint8)
            
            # Create a custom database representation using Pillow
            db_representations = self.create_pillow_database_representations(db_path)
            
            if not db_representations:
                self.error_occurred.emit("No valid images found in database or failed to process database images")
                return [], pd.DataFrame()
            
            # Use DeepFace.find with numpy array input
            try:
                dfs = DeepFace.find(
                    img_path=aligned_image_np,
                    db_path=db_path,
                    model_name="ArcFace",
                    distance_metric="cosine",
                    detector_backend="retinaface",
                    enforce_detection=False,
                    silent=True,
                    align=True
                )
            except Exception as e:
                # If DeepFace.find fails, try alternative approach
                self.error_occurred.emit(f"DeepFace.find failed: {e}. Trying alternative approach...")
                return self.process_with_individual_verification(aligned_image_np, db_path, threshold, output_folder)
            
            # Handle different return types from DeepFace.find
            if isinstance(dfs, list):
                if len(dfs) > 0:
                    valid_dfs = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty]
                    if valid_dfs:
                        combined_df = pd.concat(valid_dfs, ignore_index=True)
                    else:
                        return [], pd.DataFrame()
                else:
                    return [], pd.DataFrame()
            elif isinstance(dfs, pd.DataFrame):
                if dfs.empty:
                    return [], pd.DataFrame()
                combined_df = dfs
            else:
                return [], pd.DataFrame()
            
            # Process results
            if not combined_df.empty:
                results = []
                for index, row in combined_df.iterrows():
                    similarity_score = 1 - row['distance']
                    if similarity_score >= threshold:
                        original_filename = os.path.basename(row['identity'])
                        output_path = os.path.join(output_folder, original_filename)
                        
                        # Copy file if it doesn't already exist
                        if not os.path.exists(output_path):
                            try:
                                shutil.copy2(row['identity'], output_path)
                            except Exception as copy_error:
                                # If copy fails due to Unicode, use Pillow to save
                                try:
                                    img = Image.open(row['identity'])
                                    img.save(output_path)
                                except Exception as save_error:
                                    self.error_occurred.emit(f"Failed to copy/save {original_filename}: {save_error}")
                                    continue
                        
                        results.append({
                            'image_path': row['identity'], 
                            'score': similarity_score
                        })
                
                # Save CSV results
                if results:
                    csv_path = os.path.join(output_folder, 'similarity_results.csv')
                    results_df = pd.DataFrame(results)
                    results_df.to_csv(csv_path, index=False, encoding='utf-8')
                
                return results, combined_df
            else:
                return [], pd.DataFrame()
                
        except Exception as e:
            self.error_occurred.emit(f"Error in face processing: {str(e)}")
            return [], pd.DataFrame()

    def verify_face_similarity(self, target_image_path, candidate_image_path):
        """Debug method to verify similarity between two specific images using Pillow"""
        try:
            # Load images using Pillow
            target_image = self.load_image_with_pillow(target_image_path)
            candidate_image = self.load_image_with_pillow(candidate_image_path)
            
            if target_image is None or candidate_image is None:
                return None
            
            result = DeepFace.verify(
                img1_path=target_image,
                img2_path=candidate_image,
                model_name="ArcFace",
                distance_metric="cosine",
                detector_backend="retinaface",
                enforce_detection=False,
                align=True
            )
            return result
        except Exception as e:
            print(f"Verification error: {e}")
            return None

    def preprocess_image_for_comparison(self, image_path_or_array):
        """Ensure consistent preprocessing for all images"""
        try:
            if isinstance(image_path_or_array, str):
                # Load image from path
                img = Image.open(image_path_or_array)
                img_array = np.array(img)
            else:
                img_array = image_path_or_array
            
            # Ensure RGB format
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # Convert RGBA to RGB
                img_array = img_array[:, :, :3]
            elif len(img_array.shape) == 2:
                # Convert grayscale to RGB
                img_array = np.stack([img_array] * 3, axis=-1)
            
            # Ensure correct data type and range
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            return img_array
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.results = []
        self.setup_ui()
        self.detect_gpu()
        
    def setup_ui(self):
        self.setWindowTitle("Advanced Face Recognition Studio")
        self.setGeometry(100, 100, 1400, 900)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8f9fa, stop:1 #e9ecef);
            }
            QLabel {
                color: #343a40;
                font-size: 14px;
            }
        """)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create splitter for main layout
        splitter = QSplitter(Qt.Horizontal)
        central_widget.setLayout(QHBoxLayout())
        central_widget.layout().addWidget(splitter)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_panel.setMaximumWidth(500)
        left_panel.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
            }
        """)
        
        # Add shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        left_panel.setGraphicsEffect(shadow)
        
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(20)
        left_layout.setContentsMargins(30, 30, 30, 30)
        
        # Title
        title = QLabel("Face Recognition Studio")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #343a40;
                margin-bottom: 10px;
            }
        """)
        left_layout.addWidget(title)
        
        # Input section
        input_group = ModernGroupBox("Input Configuration")
        input_layout = QGridLayout(input_group)
        
        # Image path
        input_layout.addWidget(QLabel("Target Image:"), 0, 0)
        self.image_path_edit = ModernLineEdit("Select target image...")
        input_layout.addWidget(self.image_path_edit, 0, 1)
        self.browse_image_btn = ModernButton("Browse")
        self.browse_image_btn.clicked.connect(self.browse_image)
        input_layout.addWidget(self.browse_image_btn, 0, 2)
        
        # Database path
        input_layout.addWidget(QLabel("Database Folder:"), 1, 0)
        self.db_path_edit = ModernLineEdit("Select database folder...")
        input_layout.addWidget(self.db_path_edit, 1, 1)
        self.browse_db_btn = ModernButton("Browse")
        self.browse_db_btn.clicked.connect(self.browse_database)
        input_layout.addWidget(self.browse_db_btn, 1, 2)
        
        # Output path
        input_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        self.output_path_edit = ModernLineEdit("Select output folder...")
        input_layout.addWidget(self.output_path_edit, 2, 1)
        self.browse_output_btn = ModernButton("Browse")
        self.browse_output_btn.clicked.connect(self.browse_output)
        input_layout.addWidget(self.browse_output_btn, 2, 2)
        
        left_layout.addWidget(input_group)
        
        # Processing options
        options_group = ModernGroupBox("Processing Options")
        options_layout = QGridLayout(options_group)
        
        # Similarity threshold
        options_layout.addWidget(QLabel("Similarity Threshold:"), 0, 0)
        self.threshold_edit = ModernLineEdit("0.45")
        options_layout.addWidget(self.threshold_edit, 0, 1)
        
        # GPU/CPU selection
        options_layout.addWidget(QLabel("Processing Unit:"), 1, 0)
        self.gpu_combo = ModernComboBox()
        self.gpu_combo.addItems(["Auto-detect", "Force CPU", "Force GPU"])
        options_layout.addWidget(self.gpu_combo, 1, 1)
        
        # Batch size
        options_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 32)
        self.batch_size_spin.setValue(4)
        self.batch_size_spin.setStyleSheet("""
            QSpinBox {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
                background: white;
                color: #343a40;
            }
            QSpinBox:focus {
                border-color: #667eea;
            }
        """)
        options_layout.addWidget(self.batch_size_spin, 2, 1)
        
        # Checkboxes
        self.large_images_check = QCheckBox("Allow Large Images")
        self.large_images_check.setChecked(True)
        self.large_images_check.setStyleSheet("""
            QCheckBox {
                color: #343a40;
                font-size: 14px;
                font-weight: 500;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #e9ecef;
                border-radius: 4px;
                background: white;
            }
            QCheckBox::indicator:checked {
                background: #667eea;
                border-color: #667eea;
            }
        """)
        options_layout.addWidget(self.large_images_check, 3, 0, 1, 2)
        
        left_layout.addWidget(options_group)
        
        # Debug section
        debug_group = ModernGroupBox("Debug Tools")
        debug_layout = QVBoxLayout(debug_group)
        
        # Test verification button
        self.test_verify_btn = ModernButton("Test Face Verification")
        self.test_verify_btn.clicked.connect(self.test_face_verification)
        debug_layout.addWidget(self.test_verify_btn)
        
        left_layout.addWidget(debug_group)
        
        # Process button
        self.process_btn = ModernButton("Start Processing", primary=True)
        self.process_btn.clicked.connect(self.start_processing)
        left_layout.addWidget(self.process_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                background: #f8f9fa;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 6px;
            }
        """)
        left_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to process")
        self.status_label.setStyleSheet("color: #6c757d; font-style: italic;")
        left_layout.addWidget(self.status_label)
        
        left_layout.addStretch()
        
        # Right panel - Results
        right_panel = QWidget()
        right_panel.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
            }
        """)
        
        right_shadow = QGraphicsDropShadowEffect()
        right_shadow.setBlurRadius(20)
        right_shadow.setColor(QColor(0, 0, 0, 30))
        right_shadow.setOffset(0, 2)
        right_panel.setGraphicsEffect(right_shadow)
        
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(30, 30, 30, 30)
        
        # Results title
        results_title = QLabel("Processing Results")
        results_title.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #343a40;
                margin-bottom: 20px;
            }
        """)
        right_layout.addWidget(results_title)
        
        # Results text area
        self.results_text = QTextEdit()
        self.results_text.setStyleSheet("""
            QTextEdit {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 16px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 13px;
                background: #f8f9fa;
                color: #343a40;
            }
        """)
        right_layout.addWidget(self.results_text)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
    def detect_gpu(self):
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_info = f"GPU detected: {len(gpus)} device(s)"
                for i, gpu in enumerate(gpus):
                    gpu_info += f"\n  - {gpu.name}"
            else:
                gpu_info = "No GPU detected - will use CPU"
            
            self.results_text.append(f"🔍 Hardware Detection:\n{gpu_info}\n")
        except Exception as e:
            self.results_text.append(f"❌ GPU detection failed: {e}\n")
    
    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Target Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
    
    def browse_database(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Database Folder")
        if folder_path:
            self.db_path_edit.setText(folder_path)
    
    def browse_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_path_edit.setText(folder_path)
    
    def test_face_verification(self):
        """Test face verification between target image and first database image"""
        if not self.image_path_edit.text() or not self.db_path_edit.text():
            QMessageBox.warning(self, "Input Error", 
                              "Please select both target image and database folder first.")
            return
        
        try:
            # Find first image in database
            db_path = self.db_path_edit.text()
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
            
            test_image = None
            for root, dirs, files in os.walk(db_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        test_image = os.path.join(root, file)
                        break
                if test_image:
                    break
            
            if not test_image:
                QMessageBox.warning(self, "No Images", "No images found in database folder.")
                return
            
            self.results_text.append(f"\n🔬 Testing verification between:")
            self.results_text.append(f"Target: {os.path.basename(self.image_path_edit.text())}")
            self.results_text.append(f"Test: {os.path.basename(test_image)}")
            
            # Create a temporary processing thread for verification
            class VerificationThread(QThread):
                result_ready = Signal(dict)
                error_occurred = Signal(str)
                
                def __init__(self, target_path, test_path):
                    super().__init__()
                    self.target_path = target_path
                    self.test_path = test_path
                
                def run(self):
                    try:
                        result = DeepFace.verify(
                            img1_path=self.target_path,
                            img2_path=self.test_path,
                            model_name="ArcFace",
                            distance_metric="cosine",
                            detector_backend="retinaface",
                            enforce_detection=False,
                            align=True
                        )
                        self.result_ready.emit(result)
                    except Exception as e:
                        self.error_occurred.emit(str(e))
            
            def on_verification_result(result):
                similarity = 1 - result['distance']
                verified = result['verified']
                self.results_text.append(f"✅ Verification Result:")
                self.results_text.append(f"   Similarity: {similarity:.4f}")
                self.results_text.append(f"   Verified: {verified}")
                self.results_text.append(f"   Distance: {result['distance']:.4f}")
                self.results_text.append(f"   Threshold: {result['threshold']:.4f}\n")
            
            def on_verification_error(error):
                self.results_text.append(f"❌ Verification Error: {error}\n")
            
            self.verification_thread = VerificationThread(self.image_path_edit.text(), test_image)
            self.verification_thread.result_ready.connect(on_verification_result)
            self.verification_thread.error_occurred.connect(on_verification_error)
            self.verification_thread.start()
            
        except Exception as e:
            self.results_text.append(f"❌ Test Error: {str(e)}\n")
    
    def start_processing(self):
        # Validate inputs
        if not all([self.image_path_edit.text(), self.db_path_edit.text(), 
                   self.output_path_edit.text()]):
            QMessageBox.warning(self, "Input Error", 
                              "Please fill in all required fields.")
            return
        
        try:
            threshold = float(self.threshold_edit.text())
            if not (0 <= threshold <= 1):
                raise ValueError("Threshold must be between 0 and 1")
        except ValueError as e:
            QMessageBox.warning(self, "Input Error", 
                              f"Invalid threshold value: {e}")
            return
        
        # Prepare processing parameters
        params = {
            'image_path': self.image_path_edit.text(),
            'db_path': self.db_path_edit.text(),
            'threshold': threshold,
            'output_folder': self.output_path_edit.text(),
            'use_gpu': self.gpu_combo.currentText() != "Force CPU",
            'batch_size': self.batch_size_spin.value(),
            'allow_large_images': self.large_images_check.isChecked()
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(params['output_folder'], exist_ok=True)
        
        # Start processing thread
        self.processing_thread = ProcessingThread(params)
        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.status_updated.connect(self.update_status)
        self.processing_thread.finished_processing.connect(self.processing_finished)
        self.processing_thread.error_occurred.connect(self.processing_error)
        
        self.process_btn.setEnabled(False)
        self.process_btn.setText("Processing...")
        self.progress_bar.setValue(0)
        self.results_text.clear()
        
        self.processing_thread.start()
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_status(self, message):
        self.status_label.setText(message)
        self.results_text.append(f"📝 {message}")
    
    def processing_finished(self, results, df):
        self.results = results
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Start Processing")
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing completed!")
        
        if results:
            self.results_text.append(f"\n✅ Processing Complete!\n")
            self.results_text.append(f"Found {len(results)} similar faces:\n")
            
            # Sort results by similarity score (highest first)
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            for i, result in enumerate(sorted_results, 1):
                self.results_text.append(
                    f"{i}. {os.path.basename(result['image_path'])} "
                    f"(Similarity: {result['score']:.4f})"
                )
            
            # Export similarity matrix
            if not df.empty:
                matrix_path = os.path.join(
                    self.output_path_edit.text(), "similarity_matrix.pkl"
                )
                with open(matrix_path, "wb") as f:
                    pickle.dump(df, f)
                self.results_text.append(f"\n💾 Similarity matrix saved to: {matrix_path}")
                
                # Also save as CSV for easier viewing
                csv_matrix_path = os.path.join(
                    self.output_path_edit.text(), "similarity_matrix.csv"
                )
                df.to_csv(csv_matrix_path, index=False)
                self.results_text.append(f"💾 Similarity matrix CSV saved to: {csv_matrix_path}")
            
            # Show statistics
            if results:
                scores = [r['score'] for r in results]
                self.results_text.append(f"\n📊 Statistics:")
                self.results_text.append(f"   Highest similarity: {max(scores):.4f}")
                self.results_text.append(f"   Lowest similarity: {min(scores):.4f}")
                self.results_text.append(f"   Average similarity: {sum(scores)/len(scores):.4f}")
                
        else:
            self.results_text.append("\n❌ No similar faces found.")
            self.results_text.append("💡 Try lowering the similarity threshold or check if:")
            self.results_text.append("   - The target image contains a clear face")
            self.results_text.append("   - The database folder contains images with faces")
            self.results_text.append("   - The face detection is working properly")
    
    def processing_error(self, error_message):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("Start Processing")
        self.status_label.setText("Error occurred during processing")
        self.results_text.append(f"\n❌ Error: {error_message}")
        
        # Add troubleshooting suggestions
        self.results_text.append("\n🔧 Troubleshooting suggestions:")
        self.results_text.append("   - Check if all input paths are valid")
        self.results_text.append("   - Ensure the target image contains a face")
        self.results_text.append("   - Verify database folder contains images")
        self.results_text.append("   - Try using CPU instead of GPU")
        self.results_text.append("   - Check if you have enough memory available")
        
        QMessageBox.critical(self, "Processing Error", error_message)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Set application properties
    app.setApplicationName("Face Recognition Studio")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("Face Recognition Tools")
    
    # Create and show main window
    window = FaceRecognitionApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
