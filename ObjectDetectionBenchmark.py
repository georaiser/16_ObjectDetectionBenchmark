# %% [markdown]
# ### Advanced Tracking

# %%
# !pip install cython_bbox
# !pip install loguru
# !pip install thop
# !pip install lap

# %%
# !git clone https://github.com/georaiser/YOLOX.git
# %cd 'YOLOX'
# %ls
# !pip install -v -e .
# %cd ..

# %%
# # # !pip install git+https://github.com/openai/CLIP.git

# !pip install effdet
# !pip install deep-sort-realtime
# !pip install git+https://github.com/ultralytics/ultralytics.git@main

# %%
# Import common requirements
import os
import cv2
import numpy as np
import time
import datetime
import csv
#import clip
import json
from PIL import Image
from tqdm import tqdm

# %%
# Import Torch & Models requirements
import torch
import torchvision
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models import ResNet50_Weights
from effdet import create_model

from deep_sort_realtime.deepsort_tracker import DeepSort
from yolox.tracker.byte_tracker import BYTETracker

# %%
# Import YOLOX requirements
from yolox.exp import get_exp
from yolox.utils import postprocess

from yolox.data.data_augment import ValTransform
from yolox.utils import (fuse_model)  # Optionally fuse model layers for better performance

# %%
from ultralytics import YOLO  # YOLO11 uses the Ultralytics framework

# %%
def setup_device():
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # Use first GPU
        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")

# %%
device = setup_device()
print(device)

# %%
# COCO 92 classes (including background)
class_names_92 = ["__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eyeglasses",
"handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
"skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
"potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse",
"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hairbrush"]


# COCO 80 classes
class_names_80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# %% [markdown]
# ##### Configuration

# %%
# Model configuration
class ModelConfig:
    def __init__(self, model_name):
        self.model_name = model_name

# Video configuration
class VideoProcessorConfig:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.frame_skip = 2  # Number of frames to skip between detections
        self.tracking_algorithm = "ByteTrack"  # choose DeepSort or ByteTrack
        self.enable_tracking = True
        self.enable_counter = True
        self.enable_display = True # False for google colab
        self.enable_save = True
        self.enable_drawer = False  # False to load lines from json file and also for google colab


class YOLOXConfig:
    def __init__(self, model_name):
        self.exp_file = None  # Set to path for custom model
        self.ckpt_file = "weights/{}.pth".format(model_name)
        self.confthre = 0.5
        self.nmsthre = 0.5


class YOLOConfig:
    def __init__(self, model_name):
        self.model_path = "weights/{}.pt".format(model_name)
        self.conf_thres = 0.5
        self.iou_thres = 0.5


# Allowed classes
class AllowedClasses:
    def __init__(self, model_name):
        if "yolo" in model_name:
            self.class_names = class_names_80
        elif "faster" or "efficientdet" in model_name:
            self.class_names = class_names_92
        else:
            print("Model not supported")
            exit(0)

    def get_allowed_classes(self):
        self.allowed_classes = [
            self.class_names.index("person"),
            self.class_names.index("car"),
            self.class_names.index("truck"),
            self.class_names.index("bus"),
        ]

        # self.allowed_classes = [
        #     self.class_names.index("suitcase"),
        #     self.class_names.index("backpack"),
        #     self.class_names.index("handbag"),
        # ]

        #self.allowed_classes = list(range(0, len(self.class_names)))

        return self.allowed_classes

# %%
# Detection configuration
class DetectorConfig:
    def __init__(self, class_names, allowed_classes):
        self.class_names = class_names
        self.allowed_classes = allowed_classes
        self.threshold = 0.5
        self.iou_threshold = 0.5 # nms
        self.nms_type = "torchvision"


# Tracker configuration (DeepSort)
class TrackerConfig_DeepSort:
    def __init__(self):
        self.max_age = 10  # Frames before track deletion
        self.n_init = 3  # Frames needed to start track
        self.nms_max_overlap = 0.5  # NMS threshold
        self.max_cosine_distance = 0.3  # Feature matching threshold
        self.nn_budget = None  # Maximum feature cache size
        self.embedder = "mobilenet"  # Feature extractor. ["mobilenet","torchreid","clip_RN50","clip_RN101","clip_RN50x4",
        # "clip_RN50x16", "clip_ViT-B/32","clip_ViT-B/16","clip_ViT-L/14"]
        # print(clip.available_models())
        self.half = True  # Use half precision for embeddings and cosine distance
        self.bgr = True  # BGR input format
        self.embedder_gpu = True  # Embedder GPU acceleration
        self.today = None  # Today's date for logging

# Tracker configuration (ByteTrack)
class TrackerConfig_ByteTrack:
    def __init__(self, frame_skip):
        self.track_thresh = 0.5
        self.track_buffer = 60
        self.match_thresh = 0.6
        self.frame_rate = (
            30 / frame_skip
        )  # fps -> fps_ori/frame_skip
        self.mot20 = True
# frame_rate=10 and track_buffer=30
# -> object is tracked for 3 seconds before being removed


# Drawer configuration
class DrawerConfig:  #### REVISAR
    def __init__(self):
        self.line_thickness = 2
        self.line_type = cv2.LINE_AA
        self.color_palette = [
            (0, 0, 255),  # Red in BGR
            (255, 0, 0),  # Blue in BGR
            (255, 0, 255),  # Magenta in BGR
            (0, 165, 255),  # Orange in BGR
            (0, 255, 255),  # Yellow in BGR
            (128, 0, 128),  # Purple in BGR
            (0, 128, 128),  # Olive in BGR
            (0, 255, 0),  # Green in BGR
        ]
        self.lines = []


# Counter configuration
class CounterConfig:
    def __init__(self):
        self.text_color = (100, 10, 10)  # BGR format
        self.text_scale = 0.4
        self.text_thickness = 1
        self.line_thickness = 2
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_line_type = cv2.LINE_AA
        self.line_spacing = 25
        self.base_y_offset = 60
        self.cell_width = 40
        self.cell_height = 15
        self.background_color = (240, 240, 210)  # BGR format

# Display configuration
class DisplayConfig:  # not all parameters are used
    def __init__(self):
        # Window settings
        self.window_name = "Object Detection"
        self.window_width = 1280
        self.window_height = 720
        self.fullscreen = False

        # Box settings
        self.box_color = (0, 255, 0)  # BGR format
        self.box_thickness = 1
        self.box_line_type = cv2.LINE_AA

        # Text settings
        self.text_color = (0, 0, 0)  # BGR format
        self.text_scale = 0.4
        self.text_thickness = 1
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_line_type = cv2.LINE_AA
        self.text_padding = 5  # Pixels above box

        # Display options
        self.show_fps = True
        self.show_labels = True
        self.show_confidence = True
        self.show_tracking_id = True
        self.show_class_name = True

        # FPS display settings
        self.fps_position = (10, 30)  # (x,y) coordinates
        self.base_y_offset = 30
        self.fps_color = (0, 0, 0)
        self.fps_scale = 0.4

        # Background settings
        self.background_color = (200, 100, 200)  # BGR format


# %% [markdown]
# ##### Memory Manager

# %%
class MemoryManager:
    def __init__(self, cleanup_frequency=100):
        self.cleanup_frequency = cleanup_frequency

    def cleanup(self, frame_count):
        if frame_count % self.cleanup_frequency == 0:
            if torch.cuda.is_available():
                # Clear CUDA cache
                torch.cuda.empty_cache()
                # Force garbage collection
                import gc
                gc.collect()

    @staticmethod
    def print_memory_stats():
        if torch.cuda.is_available():
            print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# %% [markdown]
# #### Model Handler

# %% [markdown]
# ##### YOLOX Handler

# %%
class YOLOXHandler:
    """Handles YOLOX model implementation based on official demo code"""

    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device

        # Extract config parameters with defaults
        self.exp_file = kwargs.get("exp_file", None)
        self.ckpt_file = kwargs.get("ckpt_file", None)
        self.confthre = kwargs.get("confthre", 0.5)
        self.nmsthre = kwargs.get("nmsthre", 0.5)
        self.legacy = kwargs.get("legacy", False)

        # Initialize model from experiment file or model name
        self.exp = get_exp(exp_file=self.exp_file, exp_name=self.model_name)
        self.exp.test_conf = self.confthre
        self.exp.nmsthre = self.nmsthre
        self.num_classes = self.exp.num_classes
        self.test_size = self.exp.test_size

        # Load model
        self.model = self._load_model()

    def _load_model(self):
        """Load YOLOX model based on experiment definition"""
        model = self.exp.get_model()

        # Load checkpoint
        ckpt = torch.load(self.ckpt_file, map_location=self.device)

        model.load_state_dict(ckpt["model"])

        model.to(self.device)
        # Set to evaluation mode
        model.eval()

        # Optionally for better performance
        model = fuse_model(model)

        return model


# %% [markdown]
# ##### YOLO Handler

# %%
class YOLOHandler:
    """Handles YOLO11 model implementation using Ultralytics framework"""

    def __init__(self, model_name, device, **kwargs):
        self.model_name = model_name
        self.device = device

        # Extract config parameters with defaults
        self.model_path = kwargs.get("model_path", None)
        self.conf_thres = kwargs.get("conf_thres", 0.5)
        self.iou_thres = kwargs.get("iou_thres", 0.5)

        # Load model
        self.model = self._load_model()
        # Device
        self.model.to(self.device)
        # Set to evaluation mode
        self.model.eval()

    def _load_model(self):
        """Load YOLO model using Ultralytics"""
        model = YOLO(self.model_path)
        model.to(self.device)
        return model


# %% [markdown]
# ##### ClasicModel Handler

# %%
class ClasicModelHandler:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = self._load_model()

    def _load_model(self):
        """Load the specified model with appropriate configurations"""
        model = None

        if self.model_name.startswith("tf_efficientdet"):
            model = create_model(self.model_name, pretrained=True, bench_task="predict")
            model = model.to(self.device)
            model.eval()

        elif self.model_name.startswith("fasterrcnn"):
            model = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                progress=True,
                weights_backbone=ResNet50_Weights.DEFAULT,
            )
            model = model.to(self.device)
            model.eval()

            return model

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model


# %% [markdown]
# ##### MODELManager

# %%
class MODELManager:
    def __init__(self, model_name):
        #super().__init__()
        self.model_name = model_name
        self.device = setup_device()
        self.model = self._load_model()
        self.image_size = self.get_image_size()
        self.param = self.get_model_parameters(self.model)

    def _load_model(self):

        if self.model_name.startswith("tf_efficientdet"):
            model = ClasicModelHandler(self.model_name, self.device).model

        elif self.model_name.startswith("fasterrcnn"):
            model = ClasicModelHandler(self.model_name, self.device).model

        elif self.model_name.startswith("yolox"):
            yolox_config = YOLOXConfig(self.model_name)
            model = YOLOXHandler(self.model_name, self.device, **yolox_config.__dict__)

        elif self.model_name.startswith("yolov"):
            yolo_config = YOLOConfig(self.model_name)
            model = YOLOHandler(self.model_name, self.device, **yolo_config.__dict__)

        elif self.model_name.startswith("yolo11"):
            yolo_config = YOLOConfig(self.model_name)
            model = YOLOHandler(self.model_name, self.device, **yolo_config.__dict__)

        elif self.model_name.startswith("yolo12"):
            yolo_config = YOLOConfig(self.model_name)
            model = YOLOHandler(self.model_name, self.device, **yolo_config.__dict__)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        return model

    def get_model_parameters(self, model):
        """Verify and print model parameter information"""

        try:
            model_parameters = self.model.parameters()
        except AttributeError:
            model_parameters = self.model.model.parameters()

        total_params = sum(p.numel() for p in model_parameters)
        trained_params = sum(p.numel() for p in model_parameters if p.requires_grad)

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trained_params:,}")

        return total_params, trained_params

    def get_image_size(self):
        size_map = {
            "tf_efficientdet_d0": 512,
            "tf_efficientdet_d1": 640,
            "tf_efficientdet_d2": 768,
            "tf_efficientdet_d3": 896,
            "tf_efficientdet_d4": 1024,
            "tf_efficientdet_d5": 1280,
            "tf_efficientdet_d6": 1280,
            "tf_efficientdet_d7": 1536,
            "fasterrcnn_resnet50_fpn": 1024,
        }

        # If model name starts with "yolo", return 640
        if self.model_name.startswith("yolo"):
            return 640

        # Return size from size_map or default to None
        return size_map.get(self.model_name, None)

    def get_model(self):
        """Return the loaded model"""
        return self.model

    def get_model_name(self):
        """Return the loaded model name"""
        return self.model_name

    def get_device(self):
        """Return the current device"""
        return self.device

# %% [markdown]
# #### Image Processor

# %%
class ImageProcessor:

    @staticmethod
    def calculate_dimensions(original_width, original_height, target_size):
        """Calculate new dimensions maintaining aspect ratio"""
        if original_width > original_height:
            new_height = int(target_size * original_height / original_width)
            new_width = target_size
            top_pad = (target_size - new_height) // 2
            bottom_pad = target_size - new_height - top_pad
            left_pad = right_pad = 0

        else:
            new_width = int(target_size * original_width / original_height)
            new_height = target_size
            left_pad = (target_size - new_width) // 2
            right_pad = target_size - new_width - left_pad
            top_pad = bottom_pad = 0

        padding_info = {
            "top_pad": top_pad,
            "bottom_pad": bottom_pad,
            "left_pad": left_pad,
            "right_pad": right_pad,
            "original_size": (original_width, original_height),
            "resized_size": (new_width, new_height),
            "padded_size": (target_size, target_size),
        }

        return padding_info

    @staticmethod
    def pad_to_square(image, padding_info):
        if isinstance(image, Image.Image):
            # Resize PIL image to maintain aspect ratio
            resized_image = image.resize(padding_info["resized_size"])
            # Create new image with padding
            padded_image = Image.new("RGB", (padding_info["padded_size"]), (0, 0, 0))
            padded_image.paste(
                resized_image, (padding_info["left_pad"], padding_info["top_pad"])
            )
        elif isinstance(image, np.ndarray):
            # Resize numpy array image maintaining aspect ratio
            resized_image = cv2.resize(image, padding_info["resized_size"])
            # Create padded image
            padded_image = np.zeros(
                (padding_info["padded_size"][1], padding_info["padded_size"][0], 3),
                dtype=np.uint8,
            )
            # Place resized image in padded image
            padded_image[
                padding_info["top_pad"] : padding_info["top_pad"]
                + resized_image.shape[0],
                padding_info["left_pad"] : padding_info["left_pad"]
                + resized_image.shape[1],
            ] = resized_image

            # Convert numpy array to PIL image
            padded_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)

        else:
            raise TypeError("Image must be either PIL Image or NumPy array")

        return resized_image, padded_image


    @staticmethod
    def unpad_coordinates(coords, padding_info):
        """Adjusts boxes coordinates from padded space to original/resized space."""
        if len(coords) == 0:
            return coords

        # Convert numpy array to torch tensor if needed
        if isinstance(coords, np.ndarray):
            coords = torch.from_numpy(coords)

        # Extract padding information
        top_pad = padding_info["top_pad"]
        left_pad = padding_info["left_pad"]
        padded_size = padding_info["padded_size"]
        #output_size = padding_info["resized_size"]
        output_size = padding_info["original_size"]

        # Remove padding
        adjusted_coords = coords.clone()
        adjusted_coords[:, [0, 2]] -= left_pad  # x coordinates
        adjusted_coords[:, [1, 3]] -= top_pad  # y coordinates

        # Scale to resized dimensions
        scale_x = output_size[0] / (padded_size[0] - 2 * left_pad)
        scale_y = output_size[1] / (padded_size[1] - 2 * top_pad)

        adjusted_coords[:, [0, 2]] *= scale_x
        adjusted_coords[:, [1, 3]] *= scale_y

        return adjusted_coords

# %% [markdown]
# #### Detector Utilities

# %% [markdown]
# ##### NMS Filter

# %%
class NMSFilter:
    def __init__(self, iou_threshold, nms_type="torchvision"):
        self.iou_threshold = iou_threshold
        self.nms_type = nms_type

    def apply(self, boxes, scores):
        if self.nms_type == "torchvision":
            return self._nms_torchvision(boxes, scores)
        if self.nms_type == "opencv":
            return self._nms_opencv(boxes, scores)

    def _nms_torchvision(self, boxes, scores):
        keep = torchvision.ops.nms(boxes, scores, self.iou_threshold)
        return boxes[keep], scores[keep]

    def _nms_opencv(self, boxes, scores):
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.3,
            nms_threshold=self.iou_threshold,
        )

        if len(indices) > 0:
            indices = indices.flatten()
            filtered_boxes = boxes[indices]
            filtered_scores = scores[indices]
            return filtered_boxes, filtered_scores
        return np.array([]), np.array([])

# %% [markdown]
# ##### Detection Filter

# %%
class DetectionFilter:
    """Handles filtering of detections based on scores and classes"""

    def __init__(self, score_threshold, allowed_classes):
        self.score_threshold = score_threshold
        self.allowed_classes = allowed_classes

    def filter_detections(self, boxes, scores, labels):
        """Filter detections based on score threshold and allowed classes"""
        # Score threshold filtering
        keep = scores > self.score_threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # Class filtering if specified
        if self.allowed_classes:
            class_mask = torch.tensor(
                [label.item() in self.allowed_classes for label in labels],
                dtype=torch.bool,
                device=labels.device,
            )

            boxes = boxes[class_mask]
            scores = scores[class_mask]
            labels = labels[class_mask]

        return boxes, scores, labels

# %% [markdown]
# #### Model Detector

# %% [markdown]
# ##### YOLOX Detector

# %%
class YOLOXDetector:
    def __init__(self, model_handler, device, detector_config):
        self.model_handler = model_handler
        self.model = model_handler.model.model
        self.device = device
        self.detector_config = detector_config

        self.val_preproc = ValTransform(False)

        self.detection_filter = DetectionFilter(
            score_threshold=self.detector_config.threshold,
            allowed_classes=self.detector_config.allowed_classes,
        )
        self.nms_filter = NMSFilter(
            self.detector_config.iou_threshold, self.detector_config.nms_type
        )

    def detection_pipeline(self, frame, padding_info):
        # Preprocess image
        image, resized_image = self.preprocess_pad(frame, padding_info)
        # Preprocess image
        img, img_info = self.preprocess(image)
        # Inference
        boxes, scores, labels = self.inference(img, img_info)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels, frame

    def preprocess_pad(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageProcessor.pad_to_square(frame, padding_info)

        # Convert to tensor
        #img_tensor = F.to_tensor(padded_image).unsqueeze(0)

        return padded_image, resized_image

    def preprocess(self, img):
        """Preprocess image for inference"""
        img_info = {"id": 0}
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        test_size = self.model_handler.model.test_size

        ratio = min(test_size[0] / height, test_size[1] / width)
        img_info["ratio"] = ratio

        img, _ = self.val_preproc(img, None, test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float().to(self.device)

        return img, img_info

    def inference(self, img_tensor, img_info):
        """Run inference with YOLOX model"""
        with torch.no_grad():
            outputs = self.model(img_tensor)

            outputs = postprocess(
                outputs,
                self.model_handler.model.num_classes,
                self.model_handler.model.confthre,
                self.model_handler.model.nmsthre,
                class_agnostic=True
            )

        # #print(f"Min confidence: {outputs[0][:, 4].min().item()}")
        # #print(f"Max confidence: {outputs[0][:, 4].max().item()}")

        # Process output format to match the expected format in the system
        if outputs[0] is not None:
            output = outputs[0].cpu()
            bboxes = output[:, 0:4]
            # Scale back to original image dimensions
            bboxes /= img_info["ratio"]
            scores = output[:, 4] * output[:, 5]
            cls_ids = output[:, 6]

            # Format for the tracking system
            return bboxes, scores, cls_ids
        else:
            return torch.empty((0, 4)), torch.empty(0), torch.empty(0)

    def filter_detections(self, boxes, scores, labels):
        # filter threshold and classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels
        )
        # Apply NMS
        #boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageProcessor.unpad_coordinates(boxes, padding_info)
        return boxes

# %% [markdown]
# ##### YOLO Detector

# %%
class YOLODetector:
    def __init__(self, model_handler, device, detector_config):
        self.model_handler = model_handler
        self.model = model_handler.model.model
        self.device = device
        self.detector_config = detector_config

        self.detection_filter = DetectionFilter(
            score_threshold=self.detector_config.threshold,
            allowed_classes=self.detector_config.allowed_classes,
        )
        self.nms_filter = NMSFilter(
            self.detector_config.iou_threshold, self.detector_config.nms_type
        )

    def detection_pipeline(self, frame, padding_info):
        # Preprocess image
        image, resized_image = self.preprocess_pad(frame, padding_info)
         # Inference
        predictions = self.inference(image)
        # Parse detections
        boxes, scores, labels = self.parse_detections(predictions)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels, frame

    def preprocess_pad(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageProcessor.pad_to_square(frame, padding_info)

        return padded_image, resized_image

    def inference(self, img):
        # Run YOLO11 inference
        results = self.model(
            img,
            conf=self.detector_config.threshold,
            iou=self.detector_config.iou_threshold,
            verbose=False,
        )
        return results[0]

    def parse_detections(self, results):
        # Extract boxes, confidence scores, and class IDs from YOLO11 results

        boxes = results.boxes.xyxy
        scores = results.boxes.conf
        labels = results.boxes.cls
        return boxes, scores, labels

    def filter_detections(self, boxes, scores, labels):
        # Filter based on confidence and allowed classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels
        )
        # Apply NMS
        # boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageProcessor.unpad_coordinates(boxes, padding_info)
        return boxes


# %% [markdown]
# ##### FasterRCNNDetector

# %%
class FasterRCNNDetector:
    def __init__(self, model_handler, device, detector_config):
        self.model = model_handler.get_model()
        self.device = device
        self.detector_config = detector_config

        self.detection_filter = DetectionFilter(
            score_threshold=self.detector_config.threshold,
            allowed_classes=self.detector_config.allowed_classes,
        )
        self.nms_filter = NMSFilter(
            self.detector_config.iou_threshold, self.detector_config.nms_type
        )

    def detection_pipeline(self, frame, padding_info):
        # Preprocess image
        img_tensor, resized_image = self.preprocess(frame, padding_info)
        # Inference
        predictions = self.inference(img_tensor)
        # Parse detections
        boxes, scores, labels = self.parse_detections(predictions)
        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)
        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels, resized_image

    def preprocess(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageProcessor.pad_to_square(frame, padding_info)

        # Convert to tensor
        img_tensor = F.to_tensor(padded_image).unsqueeze(0)

        return img_tensor, resized_image

    def inference(self, img_tensor):
        # Ensure input tensor is on correct device
        img_tensor = img_tensor.to(self.device)
        # Ensure model is on correct device
        self.model.to(self.device)

        # model inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img_tensor)
        return predictions

    def parse_detections(self, predictions):
        boxes = predictions[0]["boxes"]
        scores = predictions[0]["scores"]
        labels = predictions[0]["labels"]
        return boxes, scores, labels

    def filter_detections(self, boxes, scores, labels):
        # filter threshold and classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels)
        # Apply NMS
        boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageProcessor.unpad_coordinates(boxes, padding_info)
        return boxes


# %% [markdown]
# ##### EfficientDetDetector

# %%
class EfficientDetDetector:
    def __init__(self, model_handler, device, detector_config):
        self.model = model_handler.get_model()
        self.device = device
        self.detector_config = detector_config

        self.detection_filter = DetectionFilter(
            score_threshold=self.detector_config.threshold,
            allowed_classes=self.detector_config.allowed_classes,
        )
        self.nms_filter = NMSFilter(
            self.detector_config.iou_threshold, self.detector_config.nms_type
        )

    def detection_pipeline(self, frame, padding_info):
        # Preprocess image
        img_tensor, resized_image = self.preprocess(frame, padding_info)
        # Inference
        predictions = self.inference(img_tensor)
        # Parse detections
        boxes, scores, labels = self.parse_detections(predictions)

        # print(f"Min confidence: {scores.min().item()}")
        # print(f"Max confidence: {scores.max().item()}")

        # Filter detections
        boxes, scores, labels = self.filter_detections(boxes, scores, labels)

        # Adjust boxes coordinates
        boxes = self.adjust_boxes(boxes, padding_info)

        return boxes, scores, labels, resized_image

    def preprocess(self, frame, padding_info):
        # Resize and pad image
        resized_image, padded_image = ImageProcessor.pad_to_square(frame, padding_info)

        # Convert to tensor and normalize
        img_tensor = F.to_tensor(padded_image).unsqueeze(0)

        # normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_tensor = F.normalize(img_tensor, mean=mean, std=std)

        return img_tensor, resized_image

    def inference(self, img_tensor):
        # Ensure input tensor is on correct device
        img_tensor = img_tensor.to(self.device)
        # Ensure model is on correct device
        self.model.to(self.device)

        # model inference
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(img_tensor)
        return predictions

    def parse_detections(self, predictions):
        boxes = predictions[0][:, :4]
        scores = predictions[0][:, 4]
        labels = predictions[0][:, 5].long()
        return boxes, scores, labels

    def filter_detections(self, boxes, scores, labels):
        # filter threshold and classes
        boxes, scores, labels = self.detection_filter.filter_detections(
            boxes, scores, labels)
        # Apply NMS
        boxes, scores = self.nms_filter.apply(boxes, scores)
        return boxes, scores, labels

    def adjust_boxes(self, boxes, padding_info):
        # Adjust boxes coordinates
        boxes = ImageProcessor.unpad_coordinates(boxes, padding_info)
        return boxes


# %% [markdown]
# ##### Detector Manager

# %%
class DetectorManager:
    def __init__(self, model_handler, detector_config):
        self.detector_config = detector_config
        self.model_handler  = model_handler

        try:
            # yolo
            self.model          = model_handler.model.model
            self.model_name     = model_handler.model.model_name
            self.device         = model_handler.model.device
        except AttributeError:
            # fasterrcnn / efficiendet
            self.model          = model_handler.get_model()
            self.model_name     = model_handler.get_model_name()
            self.device         = model_handler.get_device()

    def get_detector(self):
        detector_map = {
            "tf_efficientdet": EfficientDetDetector,
            "fasterrcnn": FasterRCNNDetector,
            "yolox": YOLOXDetector,
            "yolov5": YOLODetector,
            "yolov8": YOLODetector,
            "yolo11": YOLODetector,
            "yolo12": YOLODetector,
        }

        for key, detector_class in detector_map.items():
            if self.model_name.startswith(key):
                args = (self.model_handler, self.device, self.detector_config)
                return detector_class(*args)

        raise ValueError(f"Unsupported model: {self.model_name}")


# %% [markdown]
# #### Tracker Processor

# %%
class Tracker_DeepSort:
    def __init__(self, config):
        """Handles object tracking using DeepSORT algorithm"""
        self.tracker_config = config
        self.tracker = DeepSort(**self.tracker_config.__dict__)

    def update(self, frame, boxes, scores, labels):
        # Convert detections to DeepSORT format - [x1,y1,w,h]
        detections = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], score.item(), label.item()))

        # Update tracks
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks

# %%
class Tracker_ByteTrack:
    def __init__(self, config):
        """Handles object tracking using BYTETrack algorithm"""
        self.tracker = BYTETracker(config, config.frame_rate)
        self.track_labels = {}

    def update(self, frame, boxes, scores, labels):

        if len(boxes) == 0:
            return []

        height, width = frame.shape[:2]
        img_info = [height, width]
        img_size = [height, width]

        detections = []
        temp_labels = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            detections.append([x1, y1, x2, y2, score.item()])
            temp_labels.append(label.item())

        detections = np.array(detections)

        tracks = self.tracker.update(detections, img_info, img_size)

        # Get matching indices using scores by JRE
        temp_labels_matched = []

        scores_cpu = []
        for score in scores:
            score = score.cpu().item()
            scores_cpu.append(score)

        for track in tracks:
            track_score = track.score  # Get track score
            matched_idx = np.where(np.equal(track_score, scores_cpu))[0]

            if len(matched_idx) > 0:
                idx = matched_idx[0]
                temp_labels_matched.append(temp_labels[idx])

        for track, label in zip(tracks, temp_labels_matched):
            if track.track_id in self.track_labels:
                if track.score > self.track_labels[track.track_id][1]:
                    self.track_labels[track.track_id] = [label, track.score]
                    track.label = label
            else:
                self.track_labels[track.track_id] = [label, track.score]
                track.label = label

        return tracks


# %% [markdown]
# ##### TrackerManager

# %%
class TrackerManager:
    def __init__(self, algorithm):
        self.algorithm = algorithm

        if "DeepSort" in self.algorithm.__class__.__name__:
            self.tracker_processor = Tracker_DeepSort(self.algorithm)
        elif "ByteTrack" in self.algorithm.__class__.__name__:
            self.tracker_processor = Tracker_ByteTrack(self.algorithm)
        else:
            raise ValueError(f"Unsupported tracking algorithm: {self.algorithm}")

    def update(self, frame, boxes, scores, labels):
        return self.tracker_processor.update(frame, boxes, scores, labels)

# %% [markdown]
# #### LineDrawer

# %%
class LineDrawer:
    def __init__(self, config):
        """Initialize the LineDrawer class"""
        self.config = config

        self.video_path = config.video_config.input_path
        self.color_palette = config.drawer_config.color_palette

        self.lines_config = []
        self.frame_with_lines = None
        self.temp_start_point = None
        self.line_counter = 1

        self._load_video_frame()

    def _load_video_frame(self, frame_number=100):
        cap = cv2.VideoCapture(self.video_path)

        # Skip frames
        for _ in range(frame_number):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                break
        cap.release()

        self.frame = frame

        if not ret:
            raise RuntimeError(f"Failed to read the video: {self.video_path}")

    def _draw_line(self, event, x, y, flags, param):
        """
        Callback function to handle mouse events and draw lines.

        :param event: Mouse event.
        :param x: X-coordinate of the event.
        :param y: Y-coordinate of the event.
        :param flags: Additional flags for the event.
        :param param: Additional parameters.
        """
        if event == cv2.EVENT_LBUTTONDOWN:  # On left mouse button click
            if self.temp_start_point is None:
                # Store the first point of the line
                self.temp_start_point = (x, y)
            else:
                # Store the second point, draw the line, and reset
                temp_end_point = (x, y)

                # Use a color from the palette
                color = self.color_palette[(self.line_counter - 1) % len(self.color_palette)]

                # Define the line configuration
                line_config = {
                    'start_point': self.temp_start_point,
                    'end_point': temp_end_point,
                    'name': f'Line{self.line_counter}',
                    'color': color
                }

                self.lines_config.append(line_config)
                self.line_counter += 1

                # Draw the line on the frame
                cv2.line(self.frame, line_config['start_point'], line_config['end_point'], color,
                         self.config.drawer_config.line_thickness)
                cv2.imshow("Draw Lines", self.frame)

                # Reset the start point
                self.temp_start_point = None

    def run(self):
        """
        Start the line drawing interaction.
        """
        cv2.namedWindow("Draw Lines")
        cv2.setMouseCallback("Draw Lines", self._draw_line)

        print("Click to define points for lines (two clicks per line). Press 'q' to quit.")
        while True:
            cv2.imshow("Draw Lines", self.frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to quit
                break
            # Saving the frame
            cv2.imwrite('frame.jpg', self.frame)

        cv2.destroyAllWindows()
        # Save to JSON file
        with open("lines_geometry.json", "w") as file:
            json.dump(self.lines_config, file, indent=4)

        return self.lines_config

# %% [markdown]
# #### Counter Process

# %% [markdown]
# ##### Geometry Calculator

# %%
class GeometryCalculator:
    """Utility class for geometric calculations related to line crossing"""

    @staticmethod
    def compute_line_side(point, line_start, line_end):
        """
        Compute which side of a line a point is on.
        Returns a normalized value: positive = one side, negative = other side
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if line_length == 0:
            return 0

        # Normalized cross product
        return ((x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)) / line_length

    @staticmethod
    def is_point_near_line(point, line_start, line_end, threshold):
        """
        Check if a point is within a threshold distance of a line segment
        """
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Vector math for accurate distance calculation
        line_vec = np.array([x2 - x1, y2 - y1])
        point_vec = np.array([x - x1, y - y1])

        line_length = np.linalg.norm(line_vec)

        if line_length == 0:
            return False

        line_unit_vec = line_vec / line_length
        projection_length = np.dot(point_vec, line_unit_vec)

        # Check if projection is on the line segment
        if 0 <= projection_length <= line_length:
            distance = abs(np.cross(line_unit_vec, point_vec))
            return distance < threshold
        return False

    @staticmethod
    def get_bbox_center(bbox):
        """Calculate center point of a bounding box [x1, y1, x2, y2]"""
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    @staticmethod
    def get_adaptive_threshold(bbox):
        """Calculate an adaptive threshold based on object size"""
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return min(width, height) * 1.5  # Adjust multiplier as needed

# %% [markdown]
# ##### Line Manager

# %%
class Line:
    """Represents a counting line with associated tracking state"""
    def __init__(self, lines_geometry):
        self.start_point = lines_geometry["start_point"]
        self.end_point = lines_geometry["end_point"]
        self.name = lines_geometry.get("name") or f"Line-{id(self)}"
        self.color = lines_geometry.get("color")
        # Counter state
        self.counts = {"up": {}, "down": {}}
        self.tracked_objects = {}



# %% [markdown]
# ##### Object Counter

# %%
class ObjectCounter:
    """
    Tracks objects crossing lines and maintains counts by direction and object class
    """
    def __init__(self, allowed_classes=None, class_names=None):
        self.allowed_classes = set(allowed_classes or [])
        self.class_names = class_names or []
        self.geometry = GeometryCalculator()

    def update(self, tracks, lines):
        """
        Update object counts based on current tracked objects
        """
        for line in lines:
            self._process_line_crossings(line, tracks)

    def _process_line_crossings(self, line, tracks):
        """Process all tracks for a specific line"""
        for track in tracks:
            # Skip if not a valid track or not in allowed classes
            if not self._is_valid_track(track):
                continue

            # Get the bounding box and center point
            bbox = self._get_track_bbox(track)
            if bbox is None:
                continue

            center = self.geometry.get_bbox_center(bbox)
            threshold = self.geometry.get_adaptive_threshold(bbox)

            # Skip if not near the line
            if not self.geometry.is_point_near_line(
                center, line["start_point"], line["end_point"], threshold
            ):
                continue

            # Process the crossing
            self._update_crossing_count(line, track, center)

    def _is_valid_track(self, track):
        """Check if track is valid and belongs to allowed classes"""
        # Check if track is confirmed - different trackers use different methods
        if hasattr(track, "is_confirmed") and callable(getattr(track, "is_confirmed")):
            is_confirmed = track.is_confirmed()
        elif hasattr(track, "is_activated"):
            is_confirmed = track.is_activated
        else:
            return False

        # Check class - different trackers use different attributes
        if hasattr(track, "det_class"):
            class_id = track.det_class
        elif hasattr(track, "label"):
            class_id = track.label
        else:
            return False

        # Track must be confirmed and in allowed classes (if specified)
        return is_confirmed and (
            not self.allowed_classes or class_id in self.allowed_classes
        )

    def _get_track_bbox(self, track):
        """Get the bounding box from a track object, handling different formats"""
        if hasattr(track, "to_ltrb") and callable(getattr(track, "to_ltrb")):
            # DeepSORT format
            return track.to_ltrb()
        elif hasattr(track, "tlwh"):
            # ByteTrack format
            l, t, w, h = track.tlwh  # noqa: E741
            return [l, t, l + w, t + h]
        return None

    def _update_crossing_count(self, line, track, center):
        """Update crossing counts for a single track"""
        track_id = track.track_id

        # Get class ID according to tracker type
        class_id = track.det_class if hasattr(track, "det_class") else track.label

        # Get current side of the line
        current_side = self.geometry.compute_line_side(
            center, line["start_point"], line["end_point"]
        )

        # Initialize tracking state for new objects
        if track_id not in line["tracked_objects"]:
            line["tracked_objects"][track_id] = {
                "prev_side": current_side,
                "class": class_id,
                "counted": False,
            }
            return

        # Get the current state for this object
        current_state = line["tracked_objects"][track_id]

        # Check if the object has crossed the line
        if (
            not current_state["counted"]
            and current_state["prev_side"] * current_side <= 0
        ):
            # Determine direction of crossing (based on sign of current side)
            direction = "down" if current_side > 0 else "up"

            # Initialize counter if needed
            if class_id not in line["counts"][direction]:
                line["counts"][direction][class_id] = 0

            # Increment count
            line["counts"][direction][class_id] += 1

            # Mark as counted to prevent multiple counts for same crossing
            current_state["counted"] = True

        # Update previous side
        current_state["prev_side"] = current_side

    def get_counts(self):
        """Get counts for all lines"""
        return [
            {
                "name": line["name"],
                "up": line["counts"]["up"],
                "down": line["counts"]["down"],
                "total_up": sum(line["counts"]["up"].values()),
                "total_down": sum(line["counts"]["down"].values()),
            }
            for line in self.lines
        ]


# %% [markdown]
# ##### Counter Visualizer

# %%
class CounterVisualizer:
    """Handles visualization of counter lines and statistics"""

    def __init__(self, counter, class_names, config):
        self.counter = counter
        self.class_names = class_names
        self.config = config.counter_config

    def draw(self, frame, lines):
        """Draw all lines and statistics on the frame"""
        # Draw lines first
        for line in lines:
            cv2.line(
                frame,
                line["start_point"],
                line["end_point"],
                line["color"],
                self.config.line_thickness,
            )
        # Draw statistics
        self._draw_statistics(frame, lines)
        return frame

    def _draw_statistics(self, frame, lines):
        """Draw crossing statistics for all lines"""
        # Collect all unique classes across all lines
        all_classes = set()
        for line in lines:
            for direction in ["up", "down"]:
                all_classes.update(line["counts"][direction].keys())

        # Sort classes for consistent ordering
        sorted_classes = sorted(list(all_classes))

        # Base vertical offset
        y_offset = self.config.base_y_offset

        for line in lines:
            # Calculate totals
            total_up = sum(line["counts"]["up"].values())
            total_down = sum(line["counts"]["down"].values())
            line_summary = f"{line['name']}: Up: {total_up}, Down: {total_down}"

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                line_summary,
                self.config.text_font,
                self.config.text_scale,
                self.config.text_thickness
            )

            # Draw background
            cv2.rectangle(
                frame,
                (10, y_offset - text_height - baseline),
                (10 + text_width, y_offset + baseline),
                self.config.background_color,
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(
                frame,
                line_summary,
                (10, y_offset),
                self.config.text_font,
                self.config.text_scale,
                line["color"],
                self.config.text_thickness,
                self.config.text_line_type,
            )

            # Move to next line
            y_offset += self.config.line_spacing

            # Draw class-specific counts
            if sorted_classes:
                self._draw_class_counts(frame, line, sorted_classes, y_offset)

                # Move to next line section
                y_offset += self.config.cell_height * 3 + 10

    def _draw_class_counts(self, frame, line, classes, y_offset):
        """Draw class-specific counts for a line"""
        x_start = 10
        cell_width = self.config.cell_width
        cell_height = self.config.cell_height

        # Draw column headers (classes)
        for class_idx, class_id in enumerate(classes):
            class_name = (
                self.class_names[int(class_id)]
                if class_id < len(self.class_names)
                else f"Class {int(class_id)}"
            )

            # Get text size for proper background sizing
            (text_width, text_height), baseline = cv2.getTextSize(
                class_name,
                self.config.text_font,
                self.config.text_scale,
                self.config.text_thickness
            )

            # Draw background
            cv2.rectangle(
                frame,
                (x_start + (class_idx + 1) * cell_width, y_offset - 15),
                (x_start + (class_idx + 2) * cell_width, y_offset + baseline),
                self.config.background_color,
                cv2.FILLED,
            )

            # Draw class name
            cv2.putText(
                frame,
                class_name,
                (x_start + (class_idx + 1) * cell_width + 5, y_offset - 2),
                self.config.text_font,
                self.config.text_scale,
                (0, 0, 0),
                self.config.text_thickness,
                self.config.text_line_type,
            )

        y_offset += cell_height

        # Draw direction rows
        directions = [("Up", (0, 100, 200)), ("Down", (0, 100, 200))]

        for direction, color in directions:
            # Draw direction label up/down
            cv2.putText(
                frame,
                direction,
                (x_start, y_offset - 2),
                self.config.text_font,
                self.config.text_scale,
                color,
                self.config.text_thickness,
                self.config.text_line_type
            )

            # Draw counts for each class
            for class_idx, class_id in enumerate(classes):
                count = line["counts"][direction.lower()].get(class_id, 0)
                count_text = str(count)

                # Get text size for centering
                (text_width, _), _ = cv2.getTextSize(
                    count_text,
                    self.config.text_font,
                    self.config.text_scale,
                    self.config.text_thickness
                )

                # Draw background
                cv2.rectangle(
                    frame,
                    (x_start + (class_idx + 1) * cell_width, y_offset - cell_height),
                    (x_start + (class_idx + 2) * cell_width, y_offset),
                    self.config.background_color,
                    cv2.FILLED,
                )

                # Draw count
                text_x = (
                    x_start
                    + (class_idx + 1) * cell_width
                    + (cell_width - text_width) // 2
                )
                cv2.putText(
                    frame,
                    count_text,
                    (text_x, y_offset - 2),
                    self.config.text_font,
                    self.config.text_scale,
                    self.config.text_color,
                    self.config.text_thickness,
                    self.config.text_line_type,
                )

            # Move to next row
            y_offset += cell_height


# %% [markdown]
# ##### Counter Manager

# %%
class CounterManager:
    """Integrates object counting with the main video processing pipeline"""

    def __init__(self, class_names, allowed_classes, config):
        self.config = config
        self.counter = ObjectCounter(allowed_classes, class_names)
        self.visualizer = CounterVisualizer(self.counter, class_names, config)
        self.drawer = LineDrawer(config
                        ) if self.config.video_config.enable_drawer else None

        self.get_lines_geometry()

    def get_lines_geometry(self):
        """Get current lines geometry"""
        if self.drawer:
            self.lines_geometry = self.drawer.run()
            # Load lines from file
            self.lines_geometry = self.load_lines_geometry()
        else:
            # Load lines from file
            self.lines_geometry = self.load_lines_geometry()


    def process_frame(self, frame, tracks):
        """Process tracks for a frame and draw visualization"""
        # Update counters with current tracks
        self.counter.update(tracks, self.lines_geometry)
        # Draw visualization on the frame
        frame = self.visualizer.draw(frame, self.lines_geometry)
        return frame

    def load_lines_geometry(self):
        """
        Load lines geometry from a JSON file.
        Returns a list of line configurations.
        """
        try:
            #with open("/content/drive/MyDrive/16_AdvancedTracking/lines_geometry.json", "r") as file:
            with open("lines_geometry.json", "r") as file:
                lines = json.load(file)
                # Initialize tracking state for each line
                for line in lines:
                    line.setdefault("counts", {"up": {}, "down": {}})
                    line.setdefault("tracked_objects", {})
                return lines
        except FileNotFoundError:
            return []

    def get_counts(self):
        """Get current counts for all lines"""
        return self.counter.get_counts()


# %% [markdown]
# ##### Counter Summary

# %%
class CounterSummary:
    def __init__(self, model_handler, config):
        self.model_handler = model_handler
        self.config = config
        self.model_name = config.model_name
        self.summary_data = {}
        self.output_file = f"Summary/counter_summary_{self.model_name}.txt"
        self.start_time = None
        self.end_time = None
        self.frame_count = 0
        self.fps_measurements = []
        self.detection_counts = 0
        self.processing_stats = {
            "input_video": config.video_config.input_path,
            "output_video": config.video_config.output_path,
            "tracking_algorithm": config.video_config.tracking_algorithm
            if config.video_config.enable_tracking
            else "None",
            "frame_skip": config.video_config.frame_skip,
        }

    def start_processing(self):
        """Record the start time of processing"""
        self.start_time = datetime.datetime.now()

    def end_processing(self):
        """Record the end time of processing"""
        self.end_time = datetime.datetime.now()

    def update_frame_stats(self, fps, detection_count=0):
        """Update per-frame statistics"""
        self.frame_count += 1
        if fps > 0:  # Ignore zero FPS values
            self.fps_measurements.append(fps)
        self.detection_counts += detection_count

    def update_from_lines(self, lines_geometry):
        """Update summary data from the lines geometry data"""
        if self.model_name not in self.summary_data:
            self.summary_data[self.model_name] = {}

        for line in lines_geometry:
            line_name = line["name"]
            if line_name not in self.summary_data[self.model_name]:
                self.summary_data[self.model_name][line_name] = {
                    "up": line["counts"]["up"].copy(),
                    "down": line["counts"]["down"].copy(),
                    "total_up": sum(line["counts"]["up"].values()),
                    "total_down": sum(line["counts"]["down"].values()),
                }

    def get_processing_stats(self):
        """Calculate processing statistics"""
        stats = self.processing_stats.copy()

        # Get model parameters
        total_params, trained_params = self.model_handler.param
        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trained_params

        # Calculate timing information
        if self.start_time and self.end_time:
            processing_duration = self.end_time - self.start_time
            stats["processing_start"] = self.start_time.strftime("%Y-%m-%d %H:%M:%S")
            stats["processing_end"] = self.end_time.strftime("%Y-%m-%d %H:%M:%S")
            stats["processing_duration"] = str(processing_duration)
            stats["processing_seconds"] = processing_duration.total_seconds()

        # Calculate FPS statistics
        if self.fps_measurements:
            stats["fps_min"] = min(self.fps_measurements)
            stats["fps_max"] = max(self.fps_measurements)
            stats["fps_mean"] = sum(self.fps_measurements) / len(self.fps_measurements)
            stats["fps_median"] = sorted(self.fps_measurements)[
                len(self.fps_measurements) // 2
            ]

        # Other statistics
        stats["frames_processed"] = self.frame_count
        stats["detections_total"] = self.detection_counts
        if self.frame_count > 0:
            stats["detections_per_frame_avg"] = self.detection_counts / self.frame_count

        # Calculate total counts across all lines
        total_up = 0
        total_down = 0
        total_objects = 0

        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                total_up += counts["total_up"]
                total_down += counts["total_down"]
                total_objects += counts["total_up"] + counts["total_down"]

        stats["total_up"] = total_up
        stats["total_down"] = total_down
        stats["total_objects"] = total_objects

        # Device information
        if torch.cuda.is_available():
            stats["device"] = f"CUDA - {torch.cuda.get_device_name(0)}"
            stats["cuda_memory_allocated_peak"] = (
                f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            )
            stats["cuda_memory_reserved_peak"] = (
                f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB"
            )
        else:
            stats["device"] = "CPU"

        return stats

    def print_summary(self):
        """Print the summary to console"""
        stats = self.get_processing_stats()

        print("\n" + "=" * 80)
        print(f"OBJECT COUNTING SUMMARY FOR MODEL: {self.model_name}")
        print("=" * 80)

        # Print processing statistics
        print("\nPROCESSING INFORMATION:")
        print(f"  Input Video: {stats['input_video']}")
        print(f"  Output Video: {stats['output_video']}")
        print(f"  Model: {self.model_name}")
        print(f"  Tracking Algorithm: {stats['tracking_algorithm']}")
        print(f"  Device: {stats['device']}")
        if "cuda_memory_allocated_peak" in stats:
            print(f"  Peak GPU Memory: {stats['cuda_memory_allocated_peak']}")

        print("\nTIMING:")
        if "processing_start" in stats:
            print(f"  Start: {stats['processing_start']}")
            print(f"  End: {stats['processing_end']}")
            print(f"  Duration: {stats['processing_duration']}")
            print(f"  Total Seconds: {stats['processing_seconds']:.2f}")

        print("\nPERFORMANCE:")
        if "fps_mean" in stats:
            print(f"  Average FPS: {stats['fps_mean']:.2f}")
            print(f"  Median FPS: {stats['fps_median']:.2f}")
            print(f"  Min FPS: {stats['fps_min']:.2f}")
            print(f"  Max FPS: {stats['fps_max']:.2f}")
        print(f"  Frames Processed: {stats['frames_processed']}")
        print(f"  Frame Skip Rate: {stats['frame_skip']}")
        if "detections_per_frame_avg" in stats:
            print(
                f"  Average Detections per Frame: {stats['detections_per_frame_avg']:.2f}"
            )

        print("\nCOUNTING STATISTICS:")
        print(f"  Total Up: {stats['total_up']}")
        print(f"  Total Down: {stats['total_down']}")
        print(f"  Total Objects: {stats['total_objects']}")

        # Print detailed counting data
        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                print(f"\nLine: {line_name}")
                print("-" * 40)

                # Print class-specific counts by direction
                for direction in ["up", "down"]:
                    print(f"\n{direction.upper()} Direction:")
                    if counts[direction]:
                        for class_id, count in counts[direction].items():
                            class_name = (
                                self.config.class_names[int(class_id)]
                                if int(class_id) < len(self.config.class_names)
                                else f"Class {int(class_id)}"
                            )
                            print(f"  {class_name}: {count}")
                    else:
                        print("  No objects counted")

                # Print totals
                print(f"\nTotals:")
                print(f"  Total Up: {counts['total_up']}")
                print(f"  Total Down: {counts['total_down']}")
                print(f"  Overall Total: {counts['total_up'] + counts['total_down']}")

    def export_to_file(self):
        """Export the summary to a text file"""
        stats = self.get_processing_stats()

        with open(self.output_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write(f"OBJECT COUNTING SUMMARY FOR MODEL: {self.model_name}\n")
            f.write(f"  Total Parameters: {stats['total_parameters']}\n")
            f.write(f"  Trainable Parameters: {stats['trainable_parameters']}\n")
            f.write("=" * 80 + "\n\n")

            # Write processing statistics
            f.write("PROCESSING INFORMATION:\n")
            f.write(f"  Input Video: {stats['input_video']}\n")
            f.write(f"  Output Video: {stats['output_video']}\n")
            f.write(f"  Model: {self.model_name}\n")
            f.write(f"  Tracking Algorithm: {stats['tracking_algorithm']}\n")
            f.write(f"  Device: {stats['device']}\n")
            if "cuda_memory_allocated_peak" in stats:
                f.write(f"  Peak GPU Memory: {stats['cuda_memory_allocated_peak']}\n")

            f.write("\nTIMING:\n")
            if "processing_start" in stats:
                f.write(f"  Start: {stats['processing_start']}\n")
                f.write(f"  End: {stats['processing_end']}\n")
                f.write(f"  Duration: {stats['processing_duration']}\n")
                f.write(f"  Total Seconds: {stats['processing_seconds']:.2f}\n")

            f.write("\nPERFORMANCE:\n")
            if "fps_mean" in stats:
                f.write(f"  Average FPS: {stats['fps_mean']:.2f}\n")
                f.write(f"  Median FPS: {stats['fps_median']:.2f}\n")
                f.write(f"  Min FPS: {stats['fps_min']:.2f}\n")
                f.write(f"  Max FPS: {stats['fps_max']:.2f}\n")
            f.write(f"  Frames Processed: {stats['frames_processed']}\n")
            f.write(f"  Frame Skip Rate: {stats['frame_skip']}\n")
            if "detections_per_frame_avg" in stats:
                f.write(
                    f"  Average Detections per Frame: {stats['detections_per_frame_avg']:.2f}\n"
                )

            f.write("\nCOUNTING STATISTICS:\n")
            f.write(f"  Total Up: {stats['total_up']}\n")
            f.write(f"  Total Down: {stats['total_down']}\n")
            f.write(f"  Total Objects: {stats['total_objects']}\n")

            # Write detailed counting data
            for model, lines in self.summary_data.items():
                for line_name, counts in lines.items():
                    f.write(f"\nLine: {line_name}\n")
                    f.write("-" * 40 + "\n")

                    # Write class-specific counts by direction
                    for direction in ["up", "down"]:
                        f.write(f"\n{direction.upper()} Direction:\n")
                        if counts[direction]:
                            for class_id, count in counts[direction].items():
                                class_name = (
                                    self.config.class_names[int(class_id)]
                                    if int(class_id) < len(self.config.class_names)
                                    else f"Class {int(class_id)}"
                                )
                                f.write(f"  {class_name}: {count}\n")
                        else:
                            f.write("  No objects counted\n")

                    # Write totals
                    f.write(f"\nTotals:\n")
                    f.write(f"  Total Up: {counts['total_up']}\n")
                    f.write(f"  Total Down: {counts['total_down']}\n")
                    f.write(
                        f"  Overall Total: {counts['total_up'] + counts['total_down']}\n\n"
                    )

        #print(f"\nSummary exported to {self.output_file}")

    def export_to_csv(self):
        """Export summary data to CSV for further analysis"""
        csv_file = f"Summary/counter_summary_{self.model_name}.csv"

        # Create data for CSV
        rows = []

        # Add header row
        header = ["Model", "Line", "Direction", "Class_ID", "Class_Name", "Count"]
        rows.append(header)

        # Add data rows
        for model, lines in self.summary_data.items():
            for line_name, counts in lines.items():
                for direction in ["up", "down"]:
                    for class_id, count in counts[direction].items():
                        class_name = (
                            self.config.class_names[int(class_id)]
                            if int(class_id) < len(self.config.class_names)
                            else f"Class {int(class_id)}"
                        )
                        row = [model, line_name, direction, class_id, class_name, count]
                        rows.append(row)

        # Write to CSV
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        #print(f"CSV data exported to {csv_file}")


# %% [markdown]
# #### Display Manager

# %%
class DisplayBase:
    def __init__(self, config):
        self.config = config

    def _draw_box(self, frame, box):
        """Draw bounding box on frame"""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            self.config.box_color,
            self.config.box_thickness,
            self.config.box_line_type,
        )

        return x1, y1, x2, y2

    def _draw_label(self, frame, text, position):
        """Draw text label on frame"""
        cv2.putText(
            frame,
            text,
            position,
            self.config.text_font,
            self.config.text_scale,
            self.config.text_color,
            self.config.text_thickness,
            self.config.text_line_type,
        )

    def _draw_background(self, frame, text):
        # Base vertical offset
        y_offset = self.config.base_y_offset

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            self.config.text_font,
            self.config.text_scale,
            self.config.text_thickness
        )
        # Draw background
        cv2.rectangle(
            frame,
            (10, y_offset - text_height - baseline),
            (10 + text_width, y_offset + baseline),
            self.config.background_color,
            cv2.FILLED,
        )

    def _draw_fps(self, frame, fps):
        """Draw FPS counter on frame"""
        if self.config.show_fps:
            fps_text = f"FPS: {fps:.2f}"
            self._draw_background(frame, fps_text)
            self._draw_label(frame, fps_text, self.config.fps_position)




# %%
class DisplayManager(DisplayBase):
    def __init__(self, config, class_names):
        self.config = config
        self.class_names = class_names

    def display_frame(self, frame):
        self.window_name = self.config.window_name
        cv2.namedWindow(self.window_name)
        cv2.imshow(self.window_name, frame)
        return cv2.waitKey(1) & 0xFF

    def draw_detections(self, frame, boxes, scores, labels, fps):
        """Draw detection boxes and labels"""
        frame_out = frame.copy()

        # Draw boxes and labels
        for box, score, label in zip(boxes, scores, labels):
            # Draw box
            x1, y1, _, _ = self._draw_box(frame_out, box)
            # Draw label
            if self.config.show_labels:
                label_text = f"{self.class_names[int(label.item())]} {score:.2f}"
                self._draw_label(
                    frame_out, label_text, (x1, y1 - self.config.text_padding)
                )
        # Draw FPS
        self._draw_fps(frame_out, fps)
        return frame_out

    def draw_tracks(self, frame, tracks, fps):
        """Draw tracking boxes and labels"""
        frame_out = frame.copy()
        # Draw tracks
        for track in tracks:
            # Get track info
            track_id = track.track_id
            if hasattr(track, "to_ltrb") and callable(track.to_ltrb):
                # for DeepSORT
                if not track.is_confirmed():
                    continue
                ltrb = track.to_ltrb()
                class_id = track.det_class
            elif hasattr(track, "tlwh"):
                # for ByteTrack (STrack), convert tlwh to ltrb
                if not track.is_activated:
                    continue
                l, t, w, h = track.tlwh  # noqa: E741
                ltrb = [l, t, l + w, t + h]
                class_id = track.label
            else:
                raise AttributeError(
                    "Track object does not have 'to_ltrb()' or 'tlwh' attributes."
                )

            # Draw box
            x1, y1, _, _ = self._draw_box(frame_out, ltrb)
            # Draw label
            if self.config.show_labels:
                label_text = (
                    f"{self.class_names[int(class_id)]}-{track_id}"
                    if class_id < len(self.class_names)
                    else f"ID-{track_id}"
                )
                self._draw_label(
                    frame_out, label_text, (x1, y1 - self.config.text_padding)
                )
        # Draw FPS
        self._draw_fps(frame_out, fps)

        return frame_out

    def cleanup(self):
        """Clean up display resources"""
        cv2.destroyAllWindows()

# %% [markdown]
# #### Video IO

# %%
class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)

    def read_frame(self):
        return self.cap.read()

    def video_parameters(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fps, width, height, total_frames

    def release(self):
        self.cap.release()

# %%

class VideoWriter:
    def __init__(self, output_path, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

# %% [markdown]
# #### System / Factory

# %%
# System configuration
class SystemConfig:
    def __init__(self, model_name, input_path):
        self.model_name = model_name
        self.input_path = input_path
        self.output_path = "Output/{}_{}.mp4".format(input_path.split('/')[-1].split('.')[0], self.model_name)
        self.model_config = ModelConfig(self.model_name)
        self.video_config = VideoProcessorConfig(self.input_path, self.output_path)
        self.class_names = AllowedClasses(self.model_name).class_names
        self.allowed_classes = AllowedClasses(self.model_name).get_allowed_classes()
        self.detector_config = DetectorConfig(self.class_names, self.allowed_classes)
        self.tracker_config = {
            "DeepSort": TrackerConfig_DeepSort(),
            "ByteTrack": TrackerConfig_ByteTrack(self.video_config.frame_skip),
        }
        self.drawer_config = DrawerConfig()
        self.counter_config = CounterConfig()
        self.display_config = DisplayConfig()


# %%
class ComponentFactory:
    @staticmethod
    def create(model_handler, config):
        return {
            "detector": DetectorManager(
                model_handler, config.detector_config
            ).get_detector(),
            "tracker": TrackerManager(
                config.tracker_config[config.video_config.tracking_algorithm]
            )
            if config.video_config.enable_tracking
            else None,
            "counter": CounterManager(
                class_names=config.detector_config.class_names,
                allowed_classes=config.detector_config.allowed_classes,
                config=config,
            )
            if config.video_config.enable_counter
            else None,
            "display": DisplayManager(
                config.display_config, config.detector_config.class_names
            ),
            "memory": MemoryManager(cleanup_frequency=100),
            "summary": CounterSummary(model_handler, config),
        }

# %% [markdown]
# #### Video Processor

# %%
class VideoProcessor:
    def __init__(self, model_handler, config, max_frame):

        os.makedirs("Output", exist_ok=True)
        os.makedirs("Summary", exist_ok=True)

        self.model_handler = model_handler
        self.device = model_handler.device
        self.config = config
        self.max_frame = max_frame

        try:
            self.model_name = self.model_handler.model.model_name
        except AttributeError:
            self.model_name = self.model_handler.get_model_name()

        # Initialize components
        self.components = ComponentFactory.create(model_handler, config)
        self.detector   = self.components["detector"]
        self.tracker    = self.components["tracker"]
        self.counter    = self.components["counter"]
        self.display    = self.components["display"]
        self.memory     = self.components["memory"]
        self.summary    = self.components["summary"]

    def process_video(self, video_path, output_path):
        video_reader = VideoReader(video_path)
        target_size = self.model_handler.image_size
        frame_skip = self.config.video_config.frame_skip

        # Start summary timing
        self.summary.start_processing()

        # Get video parameters
        fps_orig, width_orig, height_orig, total_frames = video_reader.video_parameters()
        fps_adjusted = int(fps_orig / frame_skip)

        # Calculate new dimensions
        padding_info = ImageProcessor.calculate_dimensions(
            width_orig, height_orig, target_size
        )

        # Initialize video writer if needed
        writer = (
            VideoWriter(output_path, fps_adjusted, padding_info["original_size"])
            if output_path
            else None
        )

        frame_count = 0
        with tqdm(total=total_frames) as pbar:
            while True and frame_count <= self.max_frame:  # stop process by frame count
                ret, frame = video_reader.read_frame()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    # Process frame
                    start_time = time.time()

                    # Detect objects
                    if self.detector:
                        boxes, scores, labels, resized_image = (
                            self.detector.detection_pipeline(frame, padding_info)
                        )
                        # Update detection count for summary
                        self.summary.update_frame_stats(0, len(boxes))

                    # Track objects
                    if self.tracker:
                        tracks = self.tracker.update(frame, boxes, scores, labels)

                    # Draw results
                    if self.tracker:
                        frame_processed = self.display.draw_tracks(
                            frame, tracks, 1.0 / (time.time() - start_time)
                        )
                    else:
                        frame_processed = self.display.draw_detections(
                            frame,
                            boxes,
                            scores,
                            labels,
                            1.0 / (time.time() - start_time),
                        )

                    # Calculate FPS and update summary
                    current_fps = 1.0 / (time.time() - start_time)
                    self.summary.update_frame_stats(current_fps)

                    # Counter:
                    if self.counter and self.tracker:
                        frame_processed = self.counter.process_frame(
                            frame_processed, tracks
                        )

                    # Display/save results
                    if self.config.video_config.enable_display:
                        key = self.display.display_frame(frame_processed)
                        if key == ord("q"):  # Quit if 'q' is pressed
                            break

                    if writer and self.config.video_config.enable_save:
                        writer.write(frame_processed)

                frame_count += 1

                # Call memory cleanup
                self.memory.cleanup(frame_count)

                pbar.update(1)

        # End summary timing
        self.summary.end_processing()

        # Generate and export summary
        if self.counter:
            self.summary.update_from_lines(self.counter.lines_geometry)
        #self.summary.print_summary()
        self.summary.export_to_file()
        self.summary.export_to_csv()

        # Cleanup
        video_reader.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

# %%
def run():

    models_weights = [
        "tf_efficientdet_d1",
        "tf_efficientdet_d3",
        "fasterrcnn_resnet50_fpn",
        "yolox-s.pth",
        "yolox-m.pth",
        "yolov5su.pt",
        "yolov5mu.pt",
        "yolov8s.pt",
        "yolov8m.pt",
        "yolo11s.pt",
        "yolo11m.pt",
        "yolo11l.pt",
        "yolo12s.pt",
        "yolo12m.pt",
        "yolo12l.pt",
        ]
    
    models_weights = ["yolox-s.pth"]

    for model_name in models_weights:
        model_name = model_name.split(".")[0]
        print(model_name)
        config = SystemConfig(model_name, "Input/Video1.mp4")
        model_handler = MODELManager(config.model_config.model_name)
        video_processor = VideoProcessor(model_handler, config, max_frame=100)
        video_processor.process_video(
            config.video_config.input_path, config.video_config.output_path
        )
#        
run()

# %%


# %%
# import shutil
# from google.colab import files

# folder_path = '/content/Output'
# zip_path = '/content/Output.zip'

# # Comprimir la carpeta en un archivo ZIP
# shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)

# # Descargar el archivo ZIP
# files.download(zip_path)

# %%
# folder_path = '/content/Summary'
# zip_path = '/content/Summary.zip'

# # Comprimir la carpeta en un archivo ZIP
# shutil.make_archive(zip_path.replace('.zip', ''), 'zip', folder_path)

# # Descargar el archivo ZIP
# files.download(zip_path)

# %%
# ffmpeg -ss 45 -i Video1_fasterrcnn_resnet50_fpn.mp4 \
#        -ss 45 -i Video1_yolov8s.mp4 \
#        -ss 45 -i Video1_yolo11s.mp4 \
#        -ss 45 -i Video1_yolox-s.mp4 \
#        -ss 45 -i Video1_tf_efficientdet_d3.mp4 \
#        -ss 45 -i Video1_yolox-m.mp4 \
#        -filter_complex "[0:v]scale=640:360,drawtext=text='Video1_fasterrcnn':x=10:y=10:fontsize=24:fontcolor=white[v0];\
#                         [1:v]scale=640:360,drawtext=text='Video1_yolov8s':x=10:y=10:fontsize=24:fontcolor=white[v1];\
#                         [2:v]scale=640:360,drawtext=text='Video1_yolo11s':x=10:y=10:fontsize=24:fontcolor=white[v2];\
#                         [3:v]scale=640:360,drawtext=text='Video1_yolox-s':x=10:y=10:fontsize=24:fontcolor=white[v3];\
#                         [4:v]scale=640:360,drawtext=text='Video1_tf_efficientdet_d3':x=10:y=10:fontsize=24:fontcolor=white[v4];\
#                         [5:v]scale=640:360,drawtext=text='Video1_yolox-m':x=10:y=10:fontsize=24:fontcolor=white[v5];\
#                         [v0][v1]hstack[top];[v2][v3]hstack[middle];\
#                         [v4][v5]hstack[bottom];[top][middle]vstack[upper];[bottom][upper]vstack[out]" \
#        -map "[out]" -c:v libx264 -crf 23 -preset fast comparative_video_with_names.mp4


# %%
# ffmpeg -ss 45 -i Video1_fasterrcnn_resnet50_fpn.mp4 \
#        -ss 45 -i Video1_yolo11s.mp4 \
#        -ss 45 -i Video1_tf_efficientdet_d3.mp4 \
#        -ss 45 -i Video1_yolox-m.mp4 \
#        -filter_complex "[0:v]scale=640:360,drawtext=text='Video1_fasterrcnn':x=W-tw-10:y=10:fontsize=24:fontcolor=white[v0];\
#                         [1:v]scale=640:360,drawtext=text='Video1_yolo11s':x=W-tw-10:y=10:fontsize=24:fontcolor=white[v1];\
#                         [2:v]scale=640:360,drawtext=text='Video1_tf_efficientdet_d3':x=W-tw-10:y=10:fontsize=24:fontcolor=white[v2];\
#                         [3:v]scale=640:360,drawtext=text='Video1_yolox-m':x=W-tw-10:y=10:fontsize=24:fontcolor=white[v3];\
#                         [v0][v1]hstack[top];[v2][v3]hstack[bottom];\
#                         [top][bottom]vstack[out]" \
#        -map "[out]" -c:v libx264 -crf 23 -preset fast comparative_video_4_with_names_right.mp4



