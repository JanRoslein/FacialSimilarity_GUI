## Face Similarity GUI
A graphical tool for finding similar faces across image collections using deep learning.

### Overview
Face Similarity GUI allows you to find faces in your image collection that are similar to a reference face. The application uses state-of-the-art facial recognition technology to:

1. Extract and align faces from images
2. Generate facial embeddings using ArcFace
3. Compare faces using cosine similarity
4. Save similar faces to an output directory

### Requirements
- Python 3.9 is recommended and required
- DeepFace has compatibility issues with Python 3.10+

### Dependencies
- PySide6 (Qt for Python)
- DeepFace
- RetinaFace
- FAISS (both CPU version and GPU version exists - faiss-cpu and faiss-gpu)
- PyTorch
- PIL: Pillow
- NumPy (<2.0)
- Pandas

### Platform-Specific Notes
**Windows**
- TensorFlow 2.10 is the highest version supported natively on Windows
- Consult the TensorFlow website for proper installation instructions

### CUDA support requires compatible NVIDIA drivers
- Linux
- More flexible with TensorFlow versions
- Better performance with CUDA if properly configured
- Ensure your GPU drivers are up to date and compatible with your CUDA version (2.10 version works fine even with CUDA 11.2 - suited for keppler architectures and non AVX CPU architectures)

### Installation
- Create a Conda Environment (Recommended)
```conda create -n face_similarity python=3.9```
```conda activate face_similarity```

### Install Dependencies
```pip install pyside6 deepface retinaface faiss-cpu pillow numpy pandas tensoflow<2.11``` # TensorFlow 2.10 is the highest version supported natively on Windows, which also means numpy<2.0 is required

### Usage
- Run the application:
```python face_similarity_gui.py```

1. Select an input image containing a face
2. Choose a directory of images to search through
3. Set a similarity threshold (0-1) - float values, dot as decimal separator
4. Select an output directory
5. Click "Start Processing"
I
### mportant Notes
- The first run will download model weights to ~/.deepface/weights/ (Linux/macOS) or C:\Users\YourUsername\.deepface\weights\ (Windows)
- A similarity matrix is saved as a .pkl file in the output directory
- When using the same output directory with a different input image, the existing similarity matrix will be loaded by default
- If you add or remove images from your collection, you must recompute the similarity matrix

### Compiling to Binary
**Linux**
- Install Nuitka:
```pip install nuitka```

- Compile the application:
```python -m nuitka --standalone --follow-imports --enable-plugin=pyside6,numpy,torch --include-package-data=deepface,retinaface  --include-package=PIL,cv2 --include-package=pandas --include-package=numpy --include-data-files=$HOME/.deepface/weights/*.h5=.deepface/weights/ face_similarity_gui.py```

- The compiled binary will be in the face_similarity_gui.dist directory

**Windows**
- Install Nuitka:
```pip install nuitka```

- Compile the application (run in Command Prompt or PowerShell):
```python -m nuitka --standalone --follow-imports --enable-plugin=pyside6 --include-package-data=deepface,retinaface --include-package=PIL --include-package=pandas --include-package=numpy --include-package=tensorflow --include-package=cv2 --include-package=tqdm --include-package=requests --include-data-files={USER_HOME}\*.h5=.deepface/weights/ --include-data-files={conda_path}\envs\deepface\Library\bin\cudart64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cublas64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cublasLt64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\curand64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cusolver64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cusparse64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cudnn64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\cufft64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\nvrtc64_*.dll=cuda_libs/ --include-data-files={conda_path}\envs\deepface\Library\bin\nvrtc-builtins64_*.dll=cuda_libs/ face_similarity_gui.py``` # Replace {conda_path} with the path to your Conda environment and {USER_HOME} with your home directory

- For a single executable file, add the --onefile option:
```python -m nuitka --standalone --onefile --follow-imports --enable-plugin=pyside6,numpy,torch --include-package-data=deepface,retinaface --include-package=faiss  --include-package=PIL --include-package=pandas --include-package=numpy --include-data-files={USER_HOME}\.deepface\weights\*.h5=.deepface\weights\ face_similarity_gui.py```

- The compiled binary will be in the face_similarity_gui.dist directory

### Troubleshooting
*Missing Models:*
- If you encounter errors about missing model files:
1. Run the script once normally to download models
2. Check that models exist in ~/.deepface/weights/ (Linux) or C:\Users\YourUsername\.deepface\weights\ (Windows)
3. Make sure to include these files when compiling with Nuitka

*CUDA Issues:*
- Ensure NVIDIA drivers are up to date
- Check that PyTorch was installed with CUDA support: torch.cuda.is_available() should return True
- For Windows, you may need to copy CUDA DLLs to the compiled binary directory
*Memory Errors*
- Reduce batch size in the GUI
- Check available memory on your system
- For large image collections, consider processing in smaller batches