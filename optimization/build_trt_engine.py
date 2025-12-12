
import tensorrt as trt
import os
import argparse
import sys

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def build_engine(onnx_file_path, engine_file_path, precision='fp16', calibration_cache=None):
    """
    Builds a TensorRT engine from an ONNX file.
    """
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Check if file exists
    if not os.path.exists(onnx_file_path):
        print(f"ONNX file {onnx_file_path} not found.")
        sys.exit(1)
        
    # Parse ONNX
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Builder Config
    # Memory pool limit (workspace size)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30) # 2GB
    
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # We would attach the calibrator here if we had one ready
            # config.int8_calibrator = ... 
        else:
            print("Platform does not support INT8!")
            return None

    # Optimization Profiles for Dynamic Shapes
    profile = builder.create_optimization_profile()
    
    # You need to know the input tensor name. For YOLOv8 it's usually 'images'
    input_name = network.get_input(0).name
    print(f"Input tensor name: {input_name}")
    
    # Min, Opt, Max shapes
    # N, C, H, W
    profile.set_shape(input_name, (1, 3, 640, 640), (1, 3, 640, 640), (8, 3, 640, 640))
    config.add_optimization_profile(profile)

    # Build Engine
    print(f"Building {precision} engine...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine:
        with open(engine_file_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine saved to {engine_file_path}")
    else:
        print("Failed to build engine.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="../models/model.onnx", help="Path to ONNX model")
    parser.add_argument("--output", type=str, default="../models/model.engine", help="Path to save TRT engine")
    parser.add_argument("--precision", type=str, default="fp16", choices=['fp16', 'int8'], help="Precision mode")
    
    args = parser.parse_args()
    build_engine(args.onnx, args.output, args.precision)
