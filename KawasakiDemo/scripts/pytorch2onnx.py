from ultralytics import YOLO

# TODO: Specify which model you want to convert
# Model can be downloaded from https://github.com/ultralytics/ultralytics
model = YOLO("../models/0919/best.pt")
model.fuse()
model.info(verbose=False)  # Print model information
# model.export(format="onnx", opset=12) # Export the model to onnx using opset 12
model.export(format="onnx") # Export the model to onnx using opset 12
