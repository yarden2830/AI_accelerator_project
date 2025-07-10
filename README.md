# AI_accelerator_project

Running Hailo-8L on Raspberry Pi 5

## ⚙️ Prerequisites

1. Hardware
   - Raspberry Pi 5
   - Hailo-8L accelerator attached via PCIe Gen-3.0
   - Raspberry Pi Camera module

2. Software
   - Newest Pi-OS from https://www.raspberrypi.com/software/operating-systems/#raspberry-pi-os-64-bit

3. Model Files are under
  - hef_files
  - json_files
  - pytorch_files

## 🚀 Quick Start

1. Clone repo (with submodules):
   git clone --recurse-submodules https://github.com/yarden2830/AI_accelerator_project.git
   cd AI_accelerator_project

2. System update & Python:
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3 python3-pip git libcamera-apps v4l-utils
   python3 --version
   pip3 --version

3. Enable interfaces:
   - Camera: sudo raspi-config → Interface Options → Camera → Enable → Reboot
   - PCIe Gen-3.0: https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#pcie-gen-3-0

4. Running .hef model
   See “DeGirum” section below.

## 🏋️ Dataset & Training

1. Collect ~200–500 images of desired objects.
2. Label via https://www.makesense.ai/ in YOLO format.
3. Train on Colab’s GPUs:
   https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb 
4. Save `best.pt` to `pytorch_files/`.

## 🛠️ ONNX → HEF Compilation with DeGirum

1. Login: https://hub.degirum.com/login  
   (Contact khatami.mehrdad@degirum.com; mention Liran Levy & Yarden Pardo)
2. Compiler → Add New Model → DeGirum YOLO Compiler
3. Upload `pytorch_files/my_model.pt` (input 640×640)
4. Configure:
   - Runtime: HAILORT
   - Device: Hailo8L
5. Compile; wait for email.
6. Download `.hef` to `resources/hef_files/`.

## 🚗 Running Inference

1. Sanity check (CLI):
   python basic_pipelines/detection_simple.py

2. Python pipeline:
   python hailo-rpi5-examples/basic_pipelines/detection.py --hef-path= hef_files/hailo_model.hef --labels-json= json_files/labels_hailo_model.json --input=rpi -f

## 📄 License

MIT © Liran Levy & Yarden Pardo
