# ComfyUI-TensorRT-Reforge

A modernized, robust, and highly extensible TensorRT engine exporter and loader for ComfyUI.

## 💖 Acknowledgements & Origins
This project is a direct evolution of the original [ComfyUI_TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT). We want to express our deepest gratitude to the original authors and contributors. Their groundbreaking work laid the foundation for running lightning-fast TensorRT models within ComfyUI. 

**ComfyUI-TensorRT-Reforge** builds upon their incredible legacy. We have restructured the codebase to support modern PyTorch features and future-proofed the internal architecture, ensuring seamless support for next-generation diffusion models.

## ✨ Why "Reforge"? (Key Improvements)
The generative AI landscape moves incredibly fast, bringing new and structurally complex models every month. We "reforged" the original codebase to ensure high reliability and broad compatibility across different model families:

* **Next-Gen "Anima" Architecture Support**: Fully optimized for the latest **Anima** model architecture. Reforge handles its unique structural requirements during the export process, ensuring you can run this cutting-edge model with full TensorRT acceleration.
* **Dynamo-Powered ONNX Exporting**: We utilize PyTorch's Dynamo (`dynamo=True`) alongside traditional tracing. This ensures significantly higher success rates when exporting mathematically complex architectures that previously failed in older versions.
* **Dynamic Opset Management**: Automatically adjusts ONNX Opsets (e.g., 18 vs 25) based on the specific model's requirements—essential for supporting advanced features in models like Anima and Flux.

## 🚀 Supported Models
Current architecture routing officially supports:
* **Anima (New!)**
* Flux
* SD3 / SD3.5
* SDXL
* SD 1.5
* AuraFlow
* SVD (Stable Video Diffusion)

## 📦 Installation

**Requirement: This node requires a CUDA 12.x environment, with CUDA 12.8 being highly recommended for optimal performance and stability. If you are using CUDA 11, please update your drivers and toolkit. Additionally, CUDA 13.x is not currently supported at this time.**

1. Clone this repository into your `ComfyUI/custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/zaochuan5854/ComfyUI-TensorRT-Reforge.git
   ```
2. Install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
   ```
*(Note: You must have TensorRT and ONNX Runtime properly installed in your environment).*

## 🛠 Usage
1. **Exporting**: Use the "TensorRT Exporter Reforge" node. Plug in your model patcher, define your optimal profiles (batch size, width, height, context multiplier), and generate the model file.
   * **Standard Models**: Generates a `.engine` file.
   * **Anima Architecture**: Generates a `.onnx_and_engine` compound file to handle its specific graph requirements.
2. **Loading**: Use the "TensorRT Loader Reforge" node to load your newly baked engine/file from the output directory and enjoy massive performance gains.

## 🤝 Contributing & Community

### 📢 Help Us Reforge!
If you have tested a specific environment, please let us know in the Issues/Discussions using this format.

To easily collect your package versions, you can run the following command in your terminal:
```bash
python -c "import importlib.metadata; pkgs=['coloredlogs','flatbuffers','numpy','packaging','protobuf','sympy','onnx','onnxruntime-gpu','onnxscript','tensorrt-cu12','tensorrt-cu12-libs','tensorrt-cu12-bindings']; [print(f'{p}: {importlib.metadata.version(p)}') for p in pkgs if __import__('contextlib').suppress(importlib.metadata.PackageNotFoundError) or importlib.metadata.distribution(p)]"
```
*(Note: You can copy and paste the command above, or refer to the specific versions manually.)*

- **GPU**: (e.g., RTX 4090)
- **CUDA Version**: (e.g., 12.8)
- **Package Versions**: (Paste the output of the command above)
- **Result**: (Success / Specific Error Message)

### 🗣️ Join the Discussion
We are actively discussing compatibility with the upcoming CUDA 13.x and next-gen model support. Check out our Compatibility Discussion Thread to share your insights!

### 👨‍💻 Developer's Note
"Reforge" aims for maximum performance, which often means living on the bleeding edge. While I primarily test on CUDA 12.8, I'm eager to make this project robust for CUDA 13.0 and beyond with your help.

---
*Built with ❤️ for the ComfyUI community.*
