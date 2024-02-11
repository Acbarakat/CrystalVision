import cv2.dnn
import numpy as np
import pandas as pd

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

CLASSES = yaml_load(check_yaml(r".\data\labeling\YOLODataset\dataset.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_bounding_box(img, class_name, confidence, x, y, x_plus_w, y_plus_h):
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
    color = colors[CLASSES.index(class_name)]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(onnx_model, threshold=0.5):
    """
    Main function to load ONNX model, perform inference, draw bounding boxes, and display the output image.

    Args:
        onnx_model (str): Path to the ONNX model.
        input_image (str): Path to the input image.

    Returns:
        list: List of dictionaries containing detection information such as class_id, class_name, confidence, etc.
    """

    # Load the ONNX model
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    cap = cv2.VideoCapture(0)

    # Set maximum resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 9999)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 9999)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera.")
        exit()

    ret, image = cap.read()

    # print(image.shape)
    scale_w = image.shape[1] / 640.0
    scale_h = image.shape[0] / 640.0
    # print(scale_w, scale_h)

    while True:
        # Capture frame-by-frame
        ret, image = cap.read()
        if not ret:
            break

        # Preprocess the image and prepare blob for model
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1.0 / 255, size=(640, 640), swapRB=True
        )
        model.setInput(blob)

        # Perform inference
        df: pd.DataFrame = (
            pd.DataFrame(
                model.forward()[0].T, columns=["x0", "y0", "width", "height", *CLASSES]
            )
            .query(" or ".join([f"{CLASS} >= @threshold" for CLASS in CLASSES]))
            .reset_index(drop=True)
        )

        df["maxScore"] = df[CLASSES].max(axis=1)
        df["maxClass"] = df[CLASSES].idxmax(axis=1)
        df["x0"] -= df["width"] * 0.5
        df["y0"] -= df["height"] * 0.5

        result_boxes = cv2.dnn.NMSBoxes(
            df[["x0", "y0", "width", "height"]].to_numpy(),
            df["maxScore"].to_numpy(),
            threshold,
            0.45,
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
                draw_bounding_box(
                    image, detection["maxClass"], detection["maxScore"], *box
                )

        # Display the image with bounding boxes
        image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow("image", image)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
    print(cuda_devices)

    # TODO: Class-ify the main
    # TODO: Accept an image OR use the camera
    main(r"runs\detect\train\weights\best.onnx")
