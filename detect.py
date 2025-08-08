import argparse
import glob
import logging
import os
import time
import warnings
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from models import get_model, SCRFD
from utils.general import compute_euler_angles_from_rotation_matrices, draw_cube, draw_axis

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initModel(params):
    global device, face_detector, head_pose
    try:
        face_detector = SCRFD(model_path="./weights/det_10g.onnx")
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occured while loading pre-trained weights of face detection model. Exception: {e}")

    try:
        head_pose = get_model(params.network, num_classes=6, pretrained=False)
        state_dict = torch.load(params.weights, map_location=device)
        head_pose.load_state_dict(state_dict)
        logging.info("Head Pose Estimation model weights loaded.")
    except Exception as e:
        logging.info(
            f"Exception occured while loading pre-trained weights of head pose estimation model. Exception: {e}")

    head_pose.to(device)
    head_pose.eval()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation inference.')
    parser.add_argument("--network", type=str, default="resnet50", help="Model name, default `resnet50`")
    parser.add_argument(
        "--input",
        type=str,
        help="Path to input video file or camera id"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to folder with images"
    )
    parser.add_argument(
        "--draw-type",
        type=str,
        default='cube',
        choices=['cube', 'axis'],
        help="Draw cube or axis for head pose"
    )
    parser.add_argument('--weights', type=str, default="weights/resnet50.pt",
                        help='Path to head pose estimation model weights')
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")

    return parser.parse_args()


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


def expand_bbox(x_min, y_min, x_max, y_max, factor=0.2):
    """Expand the bounding box by a given factor."""
    width = x_max - x_min
    height = y_max - y_min

    x_min_new = x_min - int(factor * height)
    y_min_new = y_min - int(factor * width)
    x_max_new = x_max + int(factor * height)
    y_max_new = y_max + int(factor * width)

    return max(0, x_min_new), max(0, y_min_new), x_max_new, y_max_new


def main_video(params):
    global device
    initModel(params)
    global head_pose, face_detector

    # Initialize video capture
    video_source = params.input
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    # Initialize VideoWriter if saving video
    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                width = x_max - x_min
                x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

                image = frame[y_min:y_max, x_min:x_max]
                image = pre_process(image)
                image = image.to(device)

                start = time.time()
                rotation_matrix = head_pose(image)
                logging.info('Head pose estimation: %.2f ms' % ((time.time() - start) * 1000))

                euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
                p_pred_deg = euler[:, 0].cpu()
                y_pred_deg = euler[:, 1].cpu()
                r_pred_deg = euler[:, 2].cpu()

                if args.draw_type == "cube":
                    draw_cube(
                        frame,
                        y_pred_deg,
                        p_pred_deg,
                        r_pred_deg,
                        bbox=[x_min, y_min, x_max, y_max],
                        size=width
                    )
                else:
                    draw_axis(
                        frame,
                        y_pred_deg,
                        p_pred_deg,
                        r_pred_deg,
                        bbox=[x_min, y_min, x_max, y_max],
                        size_ratio=0.5
                    )
            # Write the frame to the video file if saving
            if params.output:
                out.write(frame)

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


def main_folder(params):
    initModel(params)
    images = sorted(glob.glob(os.path.join(params.folder, "*.jpg")))
    for image in tqdm(images):
        obj = SimpleNamespace(image=image, network=None)
        main_image(obj)


def main_image(params):
    if params.network:
        global device
        initModel(params)
        global head_pose, face_detector
    else:
        global device, head_pose, face_detector

    with torch.no_grad():
        frame = cv2.imread(params.image)
        bboxes, keypoints = face_detector.detect(frame)
        for bbox, keypoint in zip(bboxes, keypoints):
            x_min, y_min, x_max, y_max = map(int, bbox[:4])

            width = x_max - x_min
            x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max)

            image = frame[y_min:y_max, x_min:x_max]
            image = pre_process(image)
            image = image.to(device)

            start = time.time()
            rotation_matrix = head_pose(image)

            euler = np.degrees(compute_euler_angles_from_rotation_matrices(rotation_matrix))
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            print("Yaw:", y_pred_deg)
            print("Pitch:", p_pred_deg)
            print("Roll:", r_pred_deg)


if __name__ == '__main__':
    args = parse_args()
    if args.input:
        main_video(args)
    elif args.image:
        main_image(args)
    elif args.folder:
        main_folder(args)
