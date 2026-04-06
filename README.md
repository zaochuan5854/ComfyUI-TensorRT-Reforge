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

---
*Built with ❤️ for the ComfyUI community.*