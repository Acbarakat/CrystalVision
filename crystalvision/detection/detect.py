"""
Detect FFTCG cards and show them via OpenCV.
"""
import json
import logging
from pathlib import Path
from functools import partial
from typing import List, Any

import cv2
import numpy as np
import pandas as pd
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import colors


log = logging.getLogger("detector")
log.setLevel(logging.INFO)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


class Detector(QtWidgets.QMainWindow):
    """Base Detector Class"""

    def __init__(
        self,
        classes_path: str | Path,
        card_model: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.75,
    ) -> None:
        super().__init__()

        self.classes: List[str] = yaml_load(check_yaml(classes_path))["names"]
        log.info("Classes found: %s", self.classes)

        self.cap: cv2.VideoCapture = camera
        self.threshold: float = threshold
        self.iou: float = iou

        self.setWindowTitle("CrystalVision")

        # Create a QLabel to display the image
        self.label = QtWidgets.QLabel(self)
        # self.label.setGeometry(10, 10, 650, 650)

        # Create a QTimer for updating the image
        self.timer = QtCore.QTimer(self)
        if camera:
            self.timer.timeout.connect(self.run)
            self.timer.start(100)  # Update every 100 milliseconds

        self.setCentralWidget(self.label)

        self.card_model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(str(card_model))

        with Path("./data/model/multilabel.json").open("r") as lbfp:
            self.labels = json.load(lbfp)

    def detect(
        self, frame: np.ndarray, scale_w: float = 1.0, scale_h: float = 1.0
    ) -> Any:
        raise NotImplementedError()

    def render(
        self, data: Any, frame: np.ndarray, scale_w: float = 1.0, scale_h: float = 1.0
    ) -> np.ndarray:
        raise NotImplementedError()

    def run(self):
        ret, image = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to get frame")

        scale_w = image.shape[1] / 640.0  # 640 is the standard YOLOv8 input shape
        scale_h = image.shape[0] / 640.0  # 640 is the standard YOLOv8 input shape
        log.debug("Scale factors: (%s, %s)", scale_w, scale_h)

        # if cv2.waitKey(1) & 0xFF == ord("s"):
        # cv2.imwrite("saved_frame.jpg", image)
        # break

        data = self.detect(image, scale_w=scale_w, scale_h=scale_h)
        image = self.render(data, image, scale_w=scale_w, scale_h=scale_h)

        self.update_image(image)

    IMAGE_FORMAT = QtGui.QImage.Format.Format_BGR888

    def update_image(self, frame):
        image = QtGui.QImage(frame, frame.shape[1], frame.shape[0], self.IMAGE_FORMAT)

        self.label.setPixmap(QtGui.QPixmap(image))
        self.label.setScaledContents(True)  # Fit the image to the label size

    def closeEvent(self, event):
        # Perform cleanup actions before closing the window
        self.timer.stop()
        if self.cap:
            self.cap.release()  # Release the webcam
        event.accept()


class DetectYOLO(Detector):
    def __init__(
        self,
        model_path: str | Path,
        model_task: str,
        classes_path: str | Path,
        card_model: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.45,
    ) -> None:
        super().__init__(classes_path, card_model, camera, threshold, iou)

        self.model_task = model_task

        # Load YOLO weights
        log.debug("Loading the model (%s)", model_path)
        self.model: YOLO = YOLO(model_path, task=model_task)

    def detect(
        self, frame: np.ndarray, *args, **kwargs
    ) -> Any:  # pylint: disable=W0613
        return self.model.track(
            frame,
            persist=True,
            conf=self.threshold,
            device=0,
            augment=self.can_augment,
            iou=self.iou,
        )

    def forward_blobs(self, blobs, indexes):
        if len(blobs) > 1:
            self.card_model.setInput(np.concatenate(blobs))
        elif len(blobs) == 1:
            self.card_model.setInput(blobs[0])
        else:
            return pd.DataFrame(columns=self.labels)

        output = pd.DataFrame(self.card_model.forward(), columns=self.labels)
        output["index"] = indexes
        output.set_index("index", inplace=True)

        result = {}
        for key, start, end in [
            ("type_en", 0, 4),
            ("cost", 4, 15),
            ("power", 23, 34),
            ("icons", 34, 37),
            ("element_v2", 15, 23),
        ]:
            result[key] = output.iloc[:, start:end].idxmax(axis=1)

        result = pd.DataFrame(result)
        result.index = output.index
        print(result)
        return result

    def render_segment(self, data, frame):
        blobs = []
        indexes = []
        for box, mask in zip(data[0].boxes, data[0].masks):
            x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
            indexes.append(box.id.cpu().numpy().astype(int)[0])

            # Extract the sub-image within the bounding box
            sub_mask = np.zeros(frame.shape, dtype=frame.dtype)
            sub_mask = cv2.fillPoly(
                sub_mask, np.array(mask.xy, dtype=np.int32), (255, 255, 255)
            )
            masked_image = cv2.bitwise_and(frame, sub_mask)[y_min:y_max, x_min:x_max]

            cv2.imwrite(f"./runs/{self.model_task}/{indexes[-1]}.jpg", masked_image)

            blob = cv2.dnn.blobFromImage(
                masked_image, scalefactor=1.0 / 255, size=(250, 179), swapRB=True
            )
            blobs.append(np.transpose(blob, (0, 2, 3, 1)))

        return self.forward_blobs(blobs, indexes)

    def render_obb(self, data, frame):
        blobs = []
        indexes = []
        height, width = frame.shape[:2]
        for obb in data[0].obb:
            indexes.append(obb.id.cpu().numpy().astype(int)[0])
            xywhr = obb.xywhr.cpu().numpy()[0]

            # Get rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(
                (xywhr[0], xywhr[1]), xywhr[4], 1.0
            )

            # Perform the rotation
            rotated_image = cv2.warpAffine(frame, rotation_matrix, (width, height))

            x_min = x_max = xywhr[0]
            x_min -= xywhr[3] / 2
            x_max += xywhr[3] / 2

            y_min = y_max = xywhr[1]
            y_min -= xywhr[2] / 2
            y_max += xywhr[2] / 2

            sub_image = rotated_image[int(y_min) : int(y_max), int(x_min) : int(x_max)]
            cv2.imwrite(f"./runs/{self.model_task}/{indexes[-1]}.jpg", sub_image)

            blob = cv2.dnn.blobFromImage(
                sub_image, scalefactor=1.0 / 255, size=(250, 179), swapRB=True
            )
            blobs.append(np.transpose(blob, (0, 2, 3, 1)))

        return self.forward_blobs(blobs, indexes)

    def render_detect(self, data, frame):
        blobs = []
        indexes = []
        for box in data[0].boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy.cpu().numpy()[0])
            indexes.append(box.id.cpu().numpy().astype(int)[0])

            # Extract the sub-image within the bounding box
            sub_image = frame[y_min:y_max, x_min:x_max]
            cv2.imwrite(f"./runs/{self.model_task}/{indexes[-1]}.jpg", sub_image)

            blob = cv2.dnn.blobFromImage(
                sub_image, scalefactor=1.0 / 255, size=(250, 179), swapRB=True
            )
            blobs.append(np.transpose(blob, (0, 2, 3, 1)))

        return self.forward_blobs(blobs, indexes)

    def render(self, data, frame, *args, **kwargs) -> np.ndarray:
        if render_func := getattr(self, f"render_{self.model_task}"):
            render_func(data, frame)

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
        card_model: str | Path,
        camera: cv2.VideoCapture,
        threshold: float = 0.5,
        iou: float = 0.45,
    ) -> None:
        super().__init__(classes_path, card_model, camera, threshold, iou)

        # Load the ONNX model
        log.debug("Loading the model (%s)", model_path)
        # self.model = cv2.dnn.readNetFromTorch(str(model_path.with_suffix(".pt")))
        self.model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(str(model_path))
        # self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def detect(
        self, frame: np.ndarray, scale_w: float = 1.0, scale_h: float = 1.0, imgsz=640
    ) -> pd.DataFrame:
        # if frame.shape[1] > frame.shape[0]:
        #     blob_height = ((frame.shape[0] / frame.shape[1]) * imgsz)
        #     blob_width = imgsz
        # else:
        #     blob_height = imgsz
        #     blob_width = ((frame.shape[1] / frame.shape[0]) * imgsz)
        # f = imgsz / max(frame.shape)

        # blob = cv2.resize(frame, (0, 0), fy=f, fx=f)
        # blob = cv2.copyMakeBorder(blob, 0, int(imgsz - blob_height), 0, int(imgsz - blob_width), cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # print(frame.shape, blob.shape, (scale_w, scale_h))

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0 / 255, size=(imgsz, imgsz), swapRB=False
        )
        self.model.setInput(blob)

        # Perform inference
        df: pd.DataFrame = pd.DataFrame(
            self.model.forward()[0].T,
            columns=["x0", "y0", "width", "height", *self.classes],
        )
        print(df.sort_values("card").tail(10))

        df.query(
            " or ".join([f"{cls_} >= {self.threshold}" for cls_ in self.classes]),
            inplace=True,
        )
        df.reset_index(drop=True, inplace=True)

        df["maxScore"] = df[self.classes].max(axis=1)
        df["maxClass"] = df[self.classes].idxmax(axis=1)
        df["x0"] -= df["width"] * 0.5
        df["y0"] -= df["height"] * 0.5

        return df

    def render(
        self,
        df: pd.DataFrame,
        frame: np.ndarray,
        scale_w: float = 1.0,
        scale_h: float = 1.0,
    ) -> np.ndarray:
        if df.empty:
            return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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
        return cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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
    if args.plot == "dnn":
        detector = DetectDNN
    elif args.plot == "ultralytics":
        detector = DetectYOLO

    detector = partial(
        detector,
        args.model,
        args.model_task,
        args.classes,
        args.card_model,
        threshold=args.threshold,
    )

    if args.image is not None:
        # scale_w = scale_h = 1.0
        scale_w = args.image.shape[1] / 640.0  # 640 is the standard YOLOv8 input shape
        scale_h = args.image.shape[0] / 640.0  # 640 is the standard YOLOv8 input shape

        d = detector(None)
        df = d.detect(args.image, scale_h=scale_h, scale_w=scale_w)
        image = d.render(df, args.image, scale_h=scale_h, scale_w=scale_w)
        d.update_image(image)

        return d

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

    return detector(cap)


if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig()

    parser = argparse.ArgumentParser("detect")

    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument(
        "-m", "--model", required=True, type=Path, help="the YOLO model"
    )
    parser.add_argument(
        "-cm", "--card-model", required=True, type=Path, help="The custom card model"
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
    group.add_argument(
        "--camera",
        nargs=4,
        type=int,
        help="use the camera with dimesions (W, H, FPS, Cam#)",
    )
    group.add_argument("--image", type=Path, help="image path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Set logging to DEBUG"
    )

    uargs = parser.parse_args()

    if uargs.verbose:
        log.setLevel(logging.DEBUG)
        # print(cv2.getBuildInformation())
        print(uargs)

    if uargs.image:
        uargs.image = cv2.imread(str(uargs.image))

    app = QtWidgets.QApplication(sys.argv)

    app.setApplicationName("CrystalView")
    # app.setWindowIcon(newIcon("icon"))
    # app.installTranslator(translator)
    win = main(uargs)

    # if reset_config:
    #     logger.info("Resetting Qt config: %s" % win.settings.fileName())
    #     win.settings.clear()
    #     sys.exit(0)

    win.show()
    win.raise_()
    sys.exit(app.exec_())
