# AI_accelerator_project

A real-time YOLOv11s traffic-cone detector running on Raspberry Pi 5 + Hailo-8L.

## 📂 Repository Structure

- **basics/** – Core Python scripts for capture, preprocessing, and inference.  
- **gstreamer/** – Shell pipelines using GStreamer + `hailonet`.  
- **resources/** – HEF model, labels, and sample images.  
- **docs/** – Full project report, quick-start guide, figures.  
- **scripts/** – Utility scripts 

## ⚙️ Prerequisites

1. **Hardware**  
   - Raspberry Pi 5  
   - Hailo-8L accelerator  
   - CSI-2 12 MP wide-angle camera  

2. **Software**  
   - HailoRT-CLI ≥ 4.20.0  
   - GStreamer 1.0 + hailonet/hailooverlay plugins  
   - Python 3.8 venv with `pyhailort`, `opencv-python`, `numpy`

3. **Model Files**  
   - [`resources/cone_model.hef`](resources/cone_model.hef)  
   - [`resources/labels.json`](resources/labels.json)

## 🚀 Quick Start

1. Clone & enter repo:  
   ```bash
   git clone https://github.com/yarden2830/AI_accelerator_project/hailo-cone-detection
   cd hailo-cone-detection
