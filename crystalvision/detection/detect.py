"""
Detect FFTCG cards and show them via OpenCV.
"""
import logging
from pathlib import Path
from typing import List

import cv2.dnn
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import colors


log = logging.getLogger("detector")
log.setLevel(logging.INFO)


class Detector:
    """Base Detector Class"""

    def __init__(
        self,
        classes_path: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.75,
    ) -> None:
        self.classes: List[str] = yaml_load(check_yaml(classes_path))["names"]
        log.info("Classes found: %s", self.classes)

        self.cap: cv2.VideoCapture = camera
        self.threshold: float = threshold
        self.iou: float = iou

    def detect(self, frame: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
        raise NotImplementedError()

    def run(self):
        ret, image = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to get first frame")

        scale_w = image.shape[1] / 640.0  # 640 is the standard YOLOv8 input shape
        scale_h = image.shape[0] / 640.0  # 640 is the standard YOLOv8 input shape
        log.debug("Scale factors: (%s, %s)", scale_w, scale_h)

        while True:
            # Capture frame-by-frame
            ret, image = self.cap.read()
            if not ret:
                break

            if cv2.waitKey(1) & 0xFF == ord("s"):
                cv2.imwrite("saved_frame.jpg", image)
                break

            image = self.detect(image, scale_w, scale_h)

            cv2.imshow("image", image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()


class DetectYOLO(Detector):
    def __init__(
        self,
        model_path: str | Path,
        model_task: str,
        classes_path: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.45,
    ) -> None:
        super().__init__(classes_path, camera, threshold, iou)

        self.model_task = model_task

        # Load YOLO weights
        log.debug("Loading the model (%s)", model_path)
        self.model: YOLO = YOLO(model_path, task=model_task)

    def detect(self, frame: np.ndarray, *args) -> np.ndarray:  # pylint: disable=W0613
        data = self.model.track(
            frame,
            persist=True,
            conf=self.threshold,
            device=0,
            augment=self.can_augment,
            iou=self.iou,
        )

        return cv2.resize(data[0].plot(), (0, 0), fx=0.5, fy=0.5)

    @property
    def can_augment(self) -> bool:
        return self.model_task != "segment"


class DetectDNN(Detector):
    def __init__(
        self,
        model_path: str | Path,
        model_task: str,
        classes_path: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.45,
    ) -> None:
        super().__init__(classes_path, camera, threshold, iou)

        # Load the ONNX model
        log.debug("Loading the model (%s)", model_path)
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(str(model_path))
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(self, frame: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0 / 255, size=(640, 640), swapRB=True
        )
        self.model.setInput(blob)

        # Perform inference
        df: pd.DataFrame = (
            pd.DataFrame(
                self.model.forward()[0].T,
                columns=["x0", "y0", "width", "height", *self.classes],
            )
            .query(
                " or ".join([f"{cls_} >= {self.threshold}" for cls_ in self.classes])
            )
            .reset_index(drop=True)
        )

        df["maxScore"] = df[self.classes].max(axis=1)
        df["maxClass"] = df[self.classes].idxmax(axis=1)
        df["x0"] -= df["width"] * 0.5
        df["y0"] -= df["height"] * 0.5

        result_boxes = cv2.dnn.NMSBoxes(
            df[["x0", "y0", "width", "height"]].to_numpy(),
            df["maxScore"].to_numpy(),
            self.threshold,
            self.iou,
            0.5,
        )

        if len(result_boxes) > 0:
            df.query("index in @result_boxes", inplace=True)

            df[["x0", "width"]] *= scale_w
            df[["y0", "height"]] *= scale_h
            df["x1"] = df[["x0", "width"]].sum(axis=1)
            df["y1"] = df[["y0", "height"]].sum(axis=1)
            df[["x0", "y0", "x1", "y1"]] = (
                df[["x0", "y0", "x1", "y1"]].round(0).astype(int)
            )

            # Iterate through NMS results to draw bounding boxes and labels
            # TODO: Convert this to a df.apply
            for _, detection in df.iterrows():
                box = detection[["x0", "y0", "x1", "y1"]].to_numpy()
                self.draw_bounding_box(
                    frame, detection["maxClass"], detection["maxScore"], *box
                )

        # Display the image with bounding boxes
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        return frame

    def draw_bounding_box(self, img, class_name, confidence, x, y, x_plus_w, y_plus_h):
        """
        Draws bounding boxes on the input image based on the provided arguments.

        Args:
            img (numpy.ndarray): The input image to draw the bounding box on.
            class_id (int): Class ID of the detected object.
            confidence (float): Confidence score of the detected object.
            x (int): X-coordinate of the top-left corner of the bounding box.
            y (int): Y-coordinate of the top-left corner of the bounding box.
            x_plus_w (int): X-coordinate of the bottom-right corner of the bounding box.
            y_plus_h (int): Y-coordinate of the bottom-right corner of the bounding box.
        """
        label = f"{class_name} ({confidence:.2f})"
        color = colors(self.classes.index(class_name))
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(
            img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )


def main(args) -> None:
    cap = cv2.VideoCapture(args.camera[3])

    if args.camera:
        # Set resolution
        log.debug("Attempting to set camera to %s", args.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera[1])
        cap.set(cv2.CAP_PROP_FPS, args.camera[2])

        log.info(
            "Camera is (%s, %s, %s)",
            cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cap.get(cv2.CAP_PROP_FPS),
        )

    # Check if the camera is opened successfully
    if not cap.isOpened():
        log.error("Unable to open camera")
        exit()

    if args.plot == "dnn":
        detector = DetectDNN
    elif args.plot == "ultralytics":
        detector = DetectYOLO

    detector(
        args.model, args.model_task, args.classes, cap, threshold=args.threshold
    ).run()


if __name__ == "__main__":
    import argparse

    logging.basicConfig()

    parser = argparse.ArgumentParser("detect")
    parser.add_argument(
        "-m", "--model", required=True, type=Path, help="the YOLO model"
    )
    parser.add_argument(
        "-mt",
        "--model_task",
        required=True,
        help="the YOLO task",
    )
    parser.add_argument(
        "-t", "--threshold", default=0.5, type=float, help="Confidence threshold"
    )
    parser.add_argument(
        "-p",
        "--plot",
        choices=["ultralytics", "dnn"],
        default="ultralytics",
        help="Use the model for image detection",
    )
    parser.add_argument("-c", "--classes", help="The yaml file containing the classes.")
    parser.add_argument(
        "--camera",
        nargs=4,
        type=int,
        required=True,
        help="use the camera with dimesions (W, H, FPS, Cam#)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging to DEBUG"
    )

    uargs = parser.parse_args()

    if uargs.verbose:
        log.setLevel(logging.DEBUG)

        # print(cv2.getBuildInformation())

    # TODO: Accept an image OR use the camera
    main(uargs)
