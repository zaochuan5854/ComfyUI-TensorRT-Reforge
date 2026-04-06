import { app } from "../../scripts/app.js";


function inferModelType(node) {
    const modelPathWidget = node.widgets.find(w => w.name === "model_path");
    const modelTypeWidget = node.widgets.find(w => w.name === "model_type");

    if (!modelPathWidget || !modelTypeWidget || !modelPathWidget.value) return;

    const modelPath = modelPathWidget.value;
    const modelBasename = modelPath.split(/[\\/]/).pop();

    const match = modelBasename.match(/^[^.]+\.([^.]+)\.(?:engine|onnx_and_engine)$/);

    if (match) {
        const inferredModelType = match[1];
        const choices = modelTypeWidget.options?.values || [];

        if (choices.includes(inferredModelType)) {
            if (modelTypeWidget.value !== inferredModelType) {
                modelTypeWidget.value = inferredModelType;
                node.setDirtyCanvas(true);
            }
        }
    }
}

app.registerExtension({
    name: "ComfyUI-TensorRT-Reforge.ModelTypeFromPath",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "TensorRTLoaderNode") {

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);

                const modelPathWidget = this.widgets.find(w => w.name === "model_path");

                if (modelPathWidget) {
                    modelPathWidget.callback = () => {
                        inferModelType(this);
                    };
                    setTimeout(() => {
                        inferModelType(this);
                    }, 1);
                }
            };
        }
    }
});
