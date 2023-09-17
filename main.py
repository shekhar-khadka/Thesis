from openvino.runtime import Core, Model
import collections
import time
from IPython import display
from notebook_utils import *
from output_plot import *
from pre_processing import *
import collections
import time
from IPython import display
import cv2

from ultralytics import YOLO



DET_MODEL_NAME = "yolov8l"

det_model = YOLO('models/yolo_c_640.pt')

label_map = det_model.model.names
print(label_map)
det_model_path ='models/yolo_c_640_openvino_int8_model/yolo_c_640.xml'
quantized_det_model ='models/yolo_c_640_openvino_int8_model/yolo_c_640.xml'

# det_model_path ='./models/yolov8l_openvino_int8_model/yolov8l.xml'
# quantized_det_model ='./models/yolov8l_openvino_int8_model/yolov8l.xml'

core = Core()
det_ov_model = core.read_model(det_model_path)
device = "CPU"  # "GPU"
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
det_compiled_model = core.compile_model(det_ov_model, device)


def detect(image:np.ndarray, model:Model):
    """
    OpenVINO YOLOv8 model inference function. Preprocess image, runs model inference and postprocess results using NMS.
    Parameters:
        image (np.ndarray): input image.
        model (Model): OpenVINO compiled model.
    Returns:
        detections (np.ndarray): detected boxes in format [x1, y1, x2, y2, score, label]
    """
    num_outputs = len(model.outputs)
    preprocessed_image = preprocess_image(image)
    input_tensor = image_to_tensor(preprocessed_image)
    result = model(input_tensor)
    boxes = result[model.output(0)]
    masks = None
    if num_outputs > 1:
        masks = result[model.output(1)]
    input_hw = input_tensor.shape[2:]
    detections = postprocess(pred_boxes=boxes, input_hw=input_hw, orig_img=image, pred_masks=masks)
    return detections





# run_object_detection(source=VIDEO_SOURCE, flip=True, use_popup=False, model=det_ov_model, device="AUTO")
# Main processing function to run object detection.

# run_object_detection(source=VIDEO_SOURCE, flip=True, use_popup=False, model=det_ov_model, device="AUTO")
# Main processing function to run object detection.
def run_object_detection(source, flip, use_popup, model, device):
    skip_first_frames = 0
    if device != "CPU":
        model.reshape({0: [1, 3, 640, 640]})
    compiled_model = core.compile_model(model, device)
    try:
        # Create a video capture object.
        cap = cv2.VideoCapture(source)

        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(
                winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE
            )

        processing_times = collections.deque()
        while True:
            # Read the frame from the video capture.
            ret, frame = cap.read()
            if not ret:
                print("Source ended")
                break

            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(
                    src=frame,
                    dsize=None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA,
                )
            # Get the results.
            input_image = np.array(frame)

            start_time = time.time()
            # model expects RGB image, while video capturing in BGR
            detections = detect(input_image[:, :, ::-1], compiled_model)[0]
            print('************************,\n',detections)
            stop_time = time.time()

            image_with_boxes = draw_results(detections, input_image, label_map)
            frame = image_with_boxes

            processing_times.append(stop_time - start_time)
            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # Mean processing time [ms].
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time
            text = f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            # cv2.putText(
            #     img=frame,
            #     text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
            #     org=(20, 40),
            #     fontFace=cv2.FONT_HERSHEY_COMPLEX,
            #     fontScale=f_width / 1000,
            #     color=(0, 0, 255),
            #     thickness=1,
            #     lineType=cv2.LINE_AA,
            # )
            print(text)
            # Use this workaround if there is flickering.
            # if use_popup:
            print('***********',use_popup)
            # cv2.imshow(winname='title', mat=frame)
            # key = cv2.waitKey(1)
            # # escape = 27
            # if key == 27:
            #     break
            # else:
            #     # Encode numpy array to jpg.
            #     _, encoded_img = cv2.imencode(
            #         ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100]
            #     )
            #     # Create an IPython image.
            #     i = display.Image(data=encoded_img)
            #     # Display the image in this notebook.
            #     display.clear_output(wait=True)
            #     display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        cap.release()
        if use_popup:
            cv2.destroyAllWindows()



WEBCAM_INFERENCE = False

if WEBCAM_INFERENCE:
    VIDEO_SOURCE = 0  # Webcam
else:
    VIDEO_SOURCE = 'v2.mp4'
run_object_detection(source=VIDEO_SOURCE, flip=True, use_popup=False, model=det_ov_model, device="AUTO")