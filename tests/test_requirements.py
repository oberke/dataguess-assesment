import pytest
import numpy as np
import sys
import os
from unittest.mock import MagicMock, patch

# Path Setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from inference.detector import Detector
from inference.fusion import detect_drift
from inference.utils import letterbox

# Mocking TensorRT and PyCUDA for CI environments without GPU
sys.modules['tensorrt'] = MagicMock()
sys.modules['pycuda.driver'] = MagicMock()
sys.modules['pycuda.autoinit'] = MagicMock()

# Configure TRT Mock to avoid MagicMock arithmetic issues or recursion
sys.modules['tensorrt'].volume.side_effect = np.prod # properly calculate volume from shape tuple
sys.modules['tensorrt'].nptype.return_value = np.float32

class TestUnitRequirements:
    
    @pytest.fixture
    def detector(self):
        # Initialize with PyTorch backend (no mock needed for logic tests usually, but depends on weight legacy)
        # We'll use a mock model path to avoid loading failures if file missing
        with patch('inference.detector.Detector.warmup'):
            with patch('inference.detector.Detector._init_pytorch') as mock_init:
                 det = Detector(backend="pytorch", model_path="dummy.pt")
                 det.model = MagicMock() # Mock the internal Ultralytics model
                 return det

    def test_io_shape_validation(self, detector):
        """Test I/O shape consistency"""
        # Input: (480, 640, 3) image
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test Preprocess Output Shape
        blob, orig_shapes = detector.preprocess(img)
        # Expect (1, 3, 640, 640) - Batch scale
        assert blob.shape == (1, 3, 640, 640)
        assert orig_shapes[0] == (480, 640)

    def test_pre_post_process_consistency(self, detector):
        """Test consistency of pre/post processing"""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 1. Preprocess
        blob, orig_shapes = detector.preprocess(img)
        
        # 2. Mock Prediction
        from inference.utils import scale_coords
        
        # Box in 640x640
        pred_box = np.array([[0, 0, 640, 640]], dtype=np.float32)
        
        scaled = scale_coords((640, 640), pred_box, (100, 100))
        
        # Should be close to 0,0,100,100
        np.testing.assert_allclose(scaled, [[0, 0, 100, 100]], atol=1.0)

    def test_tracker_drift(self):
        """Test logic for drift detection (IoU based)"""
        # Case 1: No overlap -> Drift
        det_box = [0, 0, 100, 100]
        trk_box = [200, 200, 300, 300]
        assert detect_drift([det_box], [trk_box]) == True
        
        # Case 2: High overlap -> No Drift
        det_box2 = [0, 0, 100, 100]
        trk_box2 = [5, 5, 105, 105] 
        assert detect_drift([det_box2], [trk_box2]) == False

    def test_warmup(self):
        """Test warm-up execution"""
        with patch('inference.detector.Detector.__call__') as mock_call:
            # We bypass init logic to just test warmup call
            det = Detector.__new__(Detector) 
            det.backend = "pytorch"
            det.warmup()
            # Verify it called inference once
            assert mock_call.called

    def test_tensorrt_engine_loading(self):
        """Test TensorRT loading path (Mocked)"""
        with patch('builtins.open', create=True) as mock_open:
            with patch('tensorrt.Runtime') as mock_runtime:
                with patch('tensorrt.Logger') as mock_logger:
                    # Mock Engine and Context
                    mock_engine = MagicMock()
                    mock_context = MagicMock()
                    mock_runtime.return_value.__enter__.return_value.deserialize_cuda_engine.return_value = mock_engine
                    mock_engine.create_execution_context.return_value = mock_context
                    mock_engine.max_batch_size = 1 # Essential for size calculation
                    
                    # Mock bindings iteration to populate self.inputs/outputs
                    # Let's say we have 2 bindings: 0 (Input), 1 (Output)
                    mock_engine.__iter__.return_value = iter([0, 1])
                    # Input shape, then Output shape (YOLOv8n: 1, 84, 8400)
                    mock_engine.get_binding_shape.side_effect = [(1, 3, 640, 640), (1, 84, 8400)]
                    mock_engine.get_binding_dtype.return_value = np.float32
                    mock_engine.binding_is_input.side_effect = [True, False] # Input, then Output
                    
                    # Mock PyCUDA
                    with patch('pycuda.driver.pagelocked_empty') as mock_page:
                         # pagelocked_empty must return a real numpy array for np.copyto to work
                         # We use the size calculated by trt.volume (via side_effect=np.prod)
                         mock_page.side_effect = lambda size, dtype: np.zeros(size, dtype=dtype)
                         
                         with patch('pycuda.driver.mem_alloc') as mock_alloc:
                             with patch('pycuda.driver.Stream') as mock_stream:
                                 det = Detector(backend="tensorrt", model_path="model.engine")
                                 assert det.backend == "tensorrt"

    def test_onnx_dynamic_shapes(self):
        """
        Verify that our code allows for dynamic batching 
        (Detector accepts list of images)
        """
        det = Detector.__new__(Detector)
        det.backend = "pytorch"
        
        img1 = np.zeros((640, 640, 3), dtype=np.uint8)
        img2 = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Pass batch of 2
        blobs, _ = det.preprocess(np.array([img1, img2]))
        assert blobs.shape[0] == 2
