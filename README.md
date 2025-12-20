# YOLO Training GUI

A comprehensive PyQt5-based graphical user interface for training, evaluating, and using YOLO (You Only Look Once) object detection models.

![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![YOLO](https://img.shields.io/badge/YOLO-v8%20%7C%20v11-orange)

## Features

### 1. Dataset Configuration
- 📁 Browse and select YAML dataset files
- ⚡ Quick select from common datasets
- ✨ Create new dataset configurations with wizard
- ✏️ Edit dataset paths and class names
- 💾 Save YAML changes

### 2. Training Parameters
- 🤖 Select YOLO model (YOLOv8 or YOLOv11)
- ⚙️ Configure training hyperparameters:
  - Epochs, image size, learning rates
  - Patience for early stopping
  - Optimizer selection (Adam, AdamW, SGD, RMSProp)
- 🚀 Advanced options:
  - Pretrained weights toggle
  - Checkpoint saving intervals
  - Device selection (GPU/CPU) with auto-detection
  - Batch size configuration
- ▶️ One-click training start

### 3. Training Status
- 📊 Real-time training progress display
- ⏱️ Epoch counter and elapsed time tracking
- 📝 Log output with timestamps
- 💾 Save logs to file
- 🔄 Auto-scroll option

### 4. Model Evaluation
- 🔍 Load trained models for evaluation
- 🎚️ Adjustable confidence threshold
- 🖼️ **Single Image Evaluation:**
  - Browse and evaluate individual images
  - View detection results with bounding boxes
  - Measurement settings for specialized applications
- 📦 **Batch Evaluation:**
  - Process entire folders of images
  - Results table with performance metrics
  - Double-click to preview images

### 5. Auto Annotation
- 🏷️ Load YOLO models for auto-annotation
- 🔧 Optional SAM (Segment Anything Model) integration
- 🎯 Annotation modes: YOLO Only, SAM Only, or Hybrid
- 🔄 Navigate through images with Previous/Next
- ⚡ Auto-annotate current image or entire batch
- 💾 Save annotations in YOLO format
- ✅ Class selection for targeted annotation
- ✏️ Manual annotation editing and deletion

## Installation

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional, but recommended for training)

### Install Dependencies

```bash
# Install PyQt5
pip install PyQt5

# Install YOLO and deep learning dependencies
pip install ultralytics torch torchvision

# Install other required packages
pip install opencv-python numpy pyyaml
```

Or install all at once:

```bash
pip install PyQt5 ultralytics torch torchvision opencv-python numpy pyyaml
```

### GPU Support (Recommended)

For GPU acceleration, install CUDA-compatible PyTorch:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Running the Application

```bash
python main.py
```

### First Training Session

1. **Configure Dataset** (Dataset Configuration tab)
   - Click **Browse** to select your YAML dataset file
   - Or click **Create New** to create a new dataset configuration

2. **Set Training Parameters** (Training Parameters tab)
   - Select a model (e.g., `yolo11n.pt` for testing, `yolo11l.pt` for accuracy)
   - Set epochs (10 for testing, 100+ for production)
   - Adjust other parameters as needed

3. **Start Training** (Training Parameters tab)
   - Click **Start Training**
   - Monitor progress in the Training Status tab

4. **Evaluate Results** (Model Evaluation tab)
   - Load your trained model
   - Test on single images or batch evaluate

## Dataset Format

Your dataset should follow the YOLO format:

### Directory Structure
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### YAML Configuration
```yaml
path: ./datasets/custom
train: images/train
val: images/val
test: images/test
names:
  - class1
  - class2
  - class3
```

### Label Format (YOLO)
Each image should have a corresponding `.txt` file with the same name:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized (0-1).

## Project Structure

```
yolo-training-gui/
├── main.py                    # Application entry point
├── testing.ui                 # Qt Designer UI definition
├── ui_mainwindow.py          # Generated from testing.ui
├── ui/
│   └── pyqt/                 # PyQt5 UI components
│       ├── app.py            # Main application class
│       ├── tabs/             # Tab implementations
│       │   ├── dataset_tab.py
│       │   ├── training_tab.py
│       │   ├── status_tab.py
│       │   ├── evaluation_tab.py
│       │   └── annotation_tab.py
│       └── common/           # Reusable UI components
│           ├── file_browser.py
│           ├── image_viewer.py
│           └── ui_utils.py
├── core/                     # Business logic (framework-agnostic)
│   ├── config_manager.py
│   ├── model_manager.py
│   ├── training_manager.py
│   ├── dataset_manager.py
│   └── annotation_manager.py
├── utils/                    # Utility modules
│   ├── logging_utils.py
│   ├── event.py
│   ├── file_utils.py
│   ├── image_utils.py
│   └── yaml_utils.py
└── assets/
    └── yolo_icon.ico
```

## Architecture

### Clean Separation of Concerns

```
┌─────────────────────────────────────┐
│        UI Layer (PyQt5)             │
│  - User interaction                 │
│  - Widget management                │
│  - Qt signals/slots                 │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│     Core Layer (Framework-agnostic) │
│  - Business logic                   │
│  - Model operations                 │
│  - Training management              │
│  - Data processing                  │
└─────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of UI and Logic**: UI components delegate all business logic to core managers
2. **Event-Driven Communication**: Custom event system for loose coupling between components
3. **Reusable Components**: Common UI widgets for file browsing, image viewing, etc.
4. **Extensibility**: Easy to add new tabs or features

## UI Development

### Modifying the UI

The UI is designed using Qt Designer. To modify:

1. Open `testing.ui` in Qt Designer
2. Make your changes
3. Regenerate the Python code:
   ```bash
   pyuic5 -x testing.ui -o ui_mainwindow.py
   ```

**Important**: Never manually edit `ui_mainwindow.py` as it will be overwritten.

### 📚 Complete Guide

For a comprehensive step-by-step guide on the PyQt Designer workflow:
- **[GUIDE_UI_TO_APP.md](GUIDE_UI_TO_APP.md)** - Complete tutorial from design to application
- **[WORKFLOW_DIAGRAM.txt](WORKFLOW_DIAGRAM.txt)** - Visual workflow diagram

These guides cover:
- Using Qt Designer
- Converting UI to Python
- Understanding generated code
- Creating application logic
- Signal/slot connections
- Best practices and troubleshooting

### Adding New Features

1. Add UI elements in `testing.ui`
2. Regenerate `ui_mainwindow.py`
3. Update the relevant tab wrapper in `ui/pyqt/tabs/`
4. Add business logic to appropriate core manager in `core/`

## Tips and Best Practices

### Training Tips

- **Start Small**: Test with 5-10 epochs on a small dataset first
- **Model Selection**:
  - `yolo11n.pt`: Fastest, good for testing
  - `yolo11s.pt`: Fast, balanced
  - `yolo11m.pt`: Medium speed and accuracy
  - `yolo11l.pt`: Slower, higher accuracy
  - `yolo11x.pt`: Slowest, highest accuracy
- **Image Size**: 640 is standard, use 320 for faster training, 1280 for higher accuracy
- **Batch Size**: Adjust based on GPU memory (16 is a good starting point)

### GPU Optimization

- Use the highest batch size your GPU can handle
- Monitor GPU usage during training
- Use mixed precision training for faster performance (automatic in newer YOLO versions)

### Dataset Quality

- More diverse training data = better model
- Balance classes (similar number of examples per class)
- Use augmentation (handled automatically by YOLO)
- Validate your dataset before training

## Troubleshooting

### "PyQt5 not found"
```bash
pip install PyQt5
```

### "CUDA out of memory"
- Reduce batch size
- Reduce image size
- Use a smaller model (e.g., yolo11n instead of yolo11l)

### "No GPU available"
- Check CUDA installation: `nvidia-smi`
- Install CUDA-compatible PyTorch (see GPU Support section)
- Application will fall back to CPU automatically

### Application won't start
- Verify you're in the project root directory
- Check all dependencies are installed
- Try running with Python explicitly: `python main.py`

### Training errors
- Verify dataset paths in YAML file
- Check that labels match images (same filenames)
- Ensure label format is correct (YOLO format)
- Check console output for specific error messages

## Advanced Features

### SAM Integration (Auto Annotation)

To use SAM for enhanced annotation:

1. Install Segment Anything:
   ```bash
   pip install segment-anything
   ```

2. Download a SAM model checkpoint:
   - [SAM Models](https://github.com/facebookresearch/segment-anything#model-checkpoints)

3. Load the SAM model in the Auto Annotation tab

4. Select "YOLO+SAM Hybrid" mode for best results

### Custom Measurement Settings

The Model Evaluation tab includes measurement settings for specialized applications:
- Microns per pixel calibration
- Block offset adjustments
- Custom judgment criteria

These are useful for industrial inspection and quality control applications.

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing architecture pattern
2. UI changes are made in Qt Designer (not directly in `ui_mainwindow.py`)
3. Business logic goes in `core/` managers
4. UI-specific code stays in `ui/pyqt/`
5. Test your changes with both CPU and GPU configurations

## License

[Your License Here]

## Acknowledgments

- Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- GUI framework: [PyQt5](https://www.riverbankcomputing.com/software/pyqt/)
- Optional SAM integration: [Segment Anything](https://github.com/facebookresearch/segment-anything)

## Support

For issues, questions, or contributions:
- Check the [Troubleshooting](#troubleshooting) section
- Review the [Quick Start](#quick-start) guide
- Ensure all dependencies are correctly installed

---

**Happy Training! 🚀**

