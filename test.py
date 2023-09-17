from pre_processing import *
import collections
import time
from IPython import display
import cv2

from ultralytics import YOLO



DET_MODEL_NAME = "yolov8l"

det_model = YOLO('/.models/yolov8l.pt')

det_validator = det_model.ValidatorClass(args=args)