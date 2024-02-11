"""
Train the card object detection
"""
from multiprocessing import freeze_support

from ultralytics import YOLO


DEFAULT_MODEL = {
    "detect": "yolov8s.pt",
    "segement": "yolov8s-seg.pt",
}


def main(args):
    # Transfer learning
    # See https://github.com/ultralytics/ultralytics?tab=readme-ov-file#models
    model = YOLO(args.model)  # load an official detection model

    results = model.train(
        data="./data/labeling/YOLODataset/dataset.yaml",
        # project="crystalvision",
        batch=-1,
        epochs=30,
        device=0,
        fliplr=0.0,
        shear=0.5,
        # translate=0.0,
        mosaic=0.0,
        # save_dir="./data/yoloruns/detect"
        # datasets_dir="./data"
        exist_ok=True,
    )

    # Evaluate the model's performance on the validation set
    results = model.val(save=False)

    # Perform object detection on an image using the model
    results = model(
        [
            "https://fftcg.cdn.sewest.net/2023-07/rufus-desktop.jpg",
            "https://i.ebayimg.com/images/g/mT8AAOSw9BFkv~qU/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/usoAAOSwHc1hpY1Y/s-l1600.jpg",
        ]
    )
    for result in results:
        print(result.boxes)

    success = model.export(format="onnx", opset=12, device=0, simplify=True)
    print(success)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("train")
    parser.add_argument("-m", "--model", help="the YOLO model.")
    parser.add_argument(
        "-t",
        "--task",
        choices=["detect", "segment"],
        default="detect",
        help="Model training task",
    )

    args = parser.parse_args()
    if not args.model:
        args.model = DEFAULT_MODEL[args.task]

    print(args)

    freeze_support()

    main(args)
