import os
import numpy as np 
import cv2
from hailo_sdk_client import ClientRunner

model_name = "savor_v1"
onnx_path = "best.onnx"
calib_dir = "calib_images"
input_shape = (640, 640) 


def get_calibration_data():
    images = []
    files = [f for f in os.listdir(calib_dir) if f.endswith('.jpg')]
    
    if len(files) == 0:
        raise FileNotFoundError(f"No .jpg images found in {calib_dir}!")

    target_count = 1024
    print(f"Found {len(files)} unique images. Loading and padding to {target_count}...")
    
    for filename in files:
        if len(images) >= target_count:
            break
            
        filepath = os.path.join(calib_dir, filename)
        img = cv2.imread(filepath)
        if img is None: continue
        
        img = cv2.resize(img, input_shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    original_count = len(images)
    while len(images) < target_count:
        needed = target_count - len(images)
        images.extend(images[:needed])
        
    print(f"Loaded {original_count} unique images. Padded with duplicates to reach {len(images)}.")
    
    return {"savor_v1/input_layer1": np.array(images).astype(np.uint8)}


def main():
    print("---------------------------------------")
    print("STEP 1: Initializing Runner...")
    runner = ClientRunner(hw_arch='hailo8')
    print("STEP 1: Done.")

    print("STEP 2: Translating ONNX (Parsing)...")
    runner.translate_onnx_model(
        onnx_path,
        model_name,
        start_node_names=["images"],
        end_node_names=[
            "/model.23/cv2.0/cv2.0.2/Conv",  
            "/model.23/cv3.0/cv3.0.2/Conv",  
            "/model.23/cv4.0/cv4.0.2/Conv",  
            "/model.23/cv2.1/cv2.1.2/Conv",
            "/model.23/cv3.1/cv3.1.2/Conv",
            "/model.23/cv4.1/cv4.1.2/Conv",
            "/model.23/cv2.2/cv2.2.2/Conv",
            "/model.23/cv3.2/cv3.2.2/Conv",
            "/model.23/cv4.2/cv4.2.2/Conv",
            "/model.23/proto/cv3/conv/Conv"], 
        net_input_shapes={"images": [1, 3, 640, 640]} 
    )
    print("STEP 2: Done.")

    print("STEP 3: Loading Script...")
    alls_script = """
normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])
context_switch_param(mode=allowed)
resources_param(max_control_utilization=1.0, max_compute_utilization=1.0, max_memory_utilization=1.0)
"""
    runner.load_model_script(alls_script)
    print("STEP 3: Done.")

    print("STEP 4: Calibration Data Load...")
    calib_data = get_calibration_data()
    print("STEP 4: Done.")
    
    print("STEP 5: Optimization (Quantization)...")
    runner.load_model_script("model_optimization_flavor(optimization_level=2)")
    
    runner.optimize(calib_data)
    print("STEP 5: Done.")

    print("Compiling to HEF binary...")
    hef = runner.compile()

    output_file = f"{model_name}.hef"
    with open(output_file, "wb") as f:
        f.write(hef)
        
    print("---------------------------------------------------------")
    print(f"SUCCESS! compiled binary saved to: {os.getcwd()}/{output_file}")
    print("---------------------------------------------------------")

if __name__ == "__main__":
    main()