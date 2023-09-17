# import cv2
# import torch
# import torchvision.transforms as transforms
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn as fasterrcnn_mobilenet_v2_fpn
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from PIL import Image
#
# # Load the pre-trained Faster R-CNN model with a MobileNetV2 backbone
# model = fasterrcnn_mobilenet_v2_fpn(pretrained=True)
# model.eval()
#
# # Set up the transformation
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# # Load the labels for the COCO dataset
# LABELS = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
#     'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
#     'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag',
#     'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#     'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
#     'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
#     'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
#     'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
# ]
#
# # Open the video capture
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Read frame from the video capture
#     ret, frame = cap.read()
#
#     # Convert frame to PIL Image
#     pil_image = Image.fromarray(frame)
#
#     # Apply the transformation
#     input_tensor = transform(pil_image)
#     input_batch = input_tensor.unsqueeze(0)
#
#     # Run the model inference
#     with torch.no_grad():
#         output = model(input_batch)
#
#     # Get the predicted bounding boxes, labels, and scores
#     boxes = output[0]['boxes'].cpu().numpy()
#     labels = output[0]['labels'].cpu().numpy()
#     scores = output[0]['scores'].cpu().numpy()
#
#     # Filter predictions with scores above a threshold
#     threshold = 0.5
#     filtered_boxes = boxes[scores > threshold]
#     filtered_labels = labels[scores > threshold]
#
#     # Convert the frame to RGB color space
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Draw bounding boxes and labels on the frame
#     for box, label in zip(filtered_boxes, filtered_labels):
#         xmin, ymin, xmax, ymax = box.astype(int)
#         cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         cv2.putText(frame_rgb, LABELS[label], (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#     # Convert the frame back to BGR color space
#     frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
#
#     # Display the frame
#     cv2.imshow("Object Detection", frame_bgr)
#
#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture and close all windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large as ssdlite320_mobilenet_v2
from PIL import Image

# Load the pre-trained SSD model with a MobileNetV2 backbone
model = ssdlite320_mobilenet_v2(pretrained=True)
model.eval()

# Set up the transformation
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
])

# Load the labels for the COCO dataset
with open("coco_labels.txt") as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the video capture
    ret, frame = cap.read()

    # Convert frame to PIL Image
    pil_image = Image.fromarray(frame)

    # Apply the transformation
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    # Run the model inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted bounding boxes, labels, and scores
    boxes = output[0]['boxes'].cpu().numpy()
    labels = output[0]['labels'].cpu().numpy()
    scores = output[0]['scores'].cpu().numpy()

    # Filter predictions with scores above a threshold
    threshold = 0.5
    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]

    # Convert the frame to RGB color space
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Draw bounding boxes and labels on the frame
    for box, label in zip(filtered_boxes, filtered_labels):
        xmin, ymin, xmax, ymax = box
        xmin *= frame.shape[1]
        ymin *= frame.shape[0]
        xmax *= frame.shape[1]
        ymax *= frame.shape[0]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        cv2.rectangle(frame_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame_rgb, str(labels[label]), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(labels)
    # Convert the frame back to BGR color space
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Display the frame
    cv2.imshow("Object Detection", frame_bgr)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
