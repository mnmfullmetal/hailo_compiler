import onnx
from onnx import shape_inference
import os

model_path = "best.onnx"
if not os.path.exists(model_path):
    print(f"ERROR: Could not find {model_path} in current directory: {os.getcwd()}")
    exit()

print(f"Loading {model_path}...")
model = onnx.load(model_path)

print("Inferring shapes (this may take a few seconds)...")
model = shape_inference.infer_shapes(model)

print(f"{'Node Name':<55} | {'Output Shape'}")
print("-" * 80)

found_nodes = 0
for node in model.graph.node:
    if node.op_type == "Conv":
        out_name = node.output[0]
        
      
        shape_str = "Unknown"
        for info in model.graph.value_info:
            if info.name == out_name:
                dims = [str(d.dim_value) for d in info.type.tensor_type.shape.dim]
                shape_str = "x".join(dims)
                break
        
        if "80x80" in shape_str or "40x40" in shape_str or "20x20" in shape_str or "160x160" in shape_str:
            print(f"{node.name:<55} | {shape_str}")
            found_nodes += 1

print("-" * 80)
if found_nodes == 0:
    print("Still found nothing? Your model might use dynamic axes (dim_value=0).")
    print("Please paste the output of: print(model.graph.node[-10:])")
else:
    print("SUCCESS: Locate the 4 nodes above (one for each size).")
    print("Proto: 160x160")
    print("Heads: 80x80, 40x40, 20x20")