"""
Train the card object detection
"""
from argparse import Namespace
from typing import Dict, Any
from multiprocessing import freeze_support

from ultralytics import YOLO


# See https://github.com/ultralytics/ultralytics?tab=readme-ov-file#models
DEFAULT_MODEL: Dict[str, str] = {
    "detect": "yolov9e.pt",
    "tune-detect": "yolov9e.pt",
    "segment": "yolov8x-seg.pt",
    "tune-segment": "yolov8x-seg.pt",
    "obb": "yolov8x-obb.pt",
    "tune-obb": "yolov8x-obb.pt",
}


def main(uargs: Namespace) -> None:
    """
    Train or tune a model based on user's arguements.

    Args:
        uargs (Namespace): User's arguments.
    """

    # Transfer learning, load an official detection model
    model = YOLO(uargs.model, task=uargs.task.replace("tune-", ""))

    model_kwargs: Dict[str, Any] = {
        "batch": uargs.batch,
        "epochs": uargs.epochs,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "freeze": 20,
        "pretrained": True,
        "scale": 0.25,
        "patience": 25,
        "exist_ok": True,
        "imgsz": [640, 640],
    }

    if uargs.task.startswith("tune"):
        results = model.tune(
            data="./data/labeling/YOLODataset/dataset.yaml",
            iterations=20,
            optimizer="auto",
            plots=False,
            verbose=False,
            use_ray=False,
            resume=False,
            task=uargs.task.replace("tune-", ""),
            space={  # key: (min, max, gain(optional))
                # 'optimizer': tune.choice(['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp']),
                "lr0": (1e-7, 1e-1),  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
                "lrf": (0.00001, 0.1),  # final OneCycleLR learning rate (lr0 * lrf)
                "momentum": (0.7, 0.98),  # SGD momentum/Adam beta1
                "weight_decay": (0.0, 0.00001),  # optimizer weight decay 5e-4
                "warmup_epochs": (0.0, 5.0),  # warmup epochs (fractions ok)
                "warmup_momentum": (0.0, 0.95),  # warmup initial momentum
                # "box": (1.0, 20.0),  # box loss gain
                "cls": (0.2, 4.0),  # cls loss gain (scale with pixels)
                # "dfl": (0.4, 6.0),  # dfl loss gain
                # "hsv_h": (0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                # "hsv_s": (0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                # "hsv_v": (0.0, 0.9),  # image HSV-Value augmentation (fraction)
                # "degrees": (0.0, 22.5),  # image rotation (+/- deg)
                # "translate": (0.0, 0.9),  # image translation (+/- fraction)
                # "scale": (0.0, 0.95),  # image scale (+/- gain)
                # "shear": (0.0, 10.0),  # image shear (+/- deg)
                # "perspective": (0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                # "flipud": (0.0, 1.0),  # image flip up-down (probability)
                # "fliplr": (0.0, 1.0),  # image flip left-right (probability)
                # "mosaic": (0.0, 1.0),  # image mixup (probability)
                # "mixup": (0.0, 1.0),  # image mixup (probability)
                # "copy_paste": (0.0, 1.0),  # segment copy-paste (probability)
            },
            **model_kwargs,
        )

        model = YOLO(f"./runs/{uargs.task.replace('tune-', '')}/tune/weights/best.pt")

        return model.export(format="onnx", device="0", simplify=True, imgsz=(640, 640))

    # model = YOLO("./runs/detect/train/weights/best.pt")
    # model = YOLO(f"./runs/{uargs.task.replace('tune-', '')}/tune/weights/best.pt")
    # return model.export(format="onnx", device="0", simplify=True)

    results = model.train(
        data="./data/labeling/YOLODataset/dataset.yaml",
        verbose=uargs.verbose,
        task=uargs.task.replace("tune-", ""),
        **model_kwargs,
        **uargs.hyperparams,
    )

    # Evaluate the model's performance on the validation set
    # results = model.val(max_det=116)

    # Perform object detection on an image using the model
    # on the "testing" set
    results = model(
        [
            "https://i.ebayimg.com/images/g/mT8AAOSw9BFkv~qU/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/usoAAOSwHc1hpY1Y/s-l1600.jpg",
            "https://i.ebayimg.com/images/g/DZEAAOSwr7BlfT9N/s-l960.jpg",
            "https://i.ebayimg.com/images/g/k5cAAOSwoSRlNMs1/s-l1600.jpg",
        ]
    )
    cls_match = ("card_foil", "crystal", "card", "card_foil_fullart")

    for result, cls_name in zip(results, cls_match):
        if uargs.verbose:
            print(result.boxes)
        assert (
            result.boxes.cls.shape[0] == 1
        ), f"Found {result.boxes.cls.shape[0]} objects"
        # assert result.names[int(result.boxes.cls.cpu()[0])] == cls_name, f"Object is '{result.names[int(result.boxes.cls.cpu()[0])]}'"
        # assert result.boxes.conf.cpu()[0] >= 0.5

    return model.export(format="onnx", device="0", simplify=True, imgsz=(640, 640))


if __name__ == "__main__":
    import yaml
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser("train")
    parser.add_argument("-m", "--model", help="the YOLO model")
    parser.add_argument(
        "-t",
        "--task",
        choices=["tune-detect", "detect", "tune-segment", "segment", "tune-obb", "obb"],
        default="tune",
        help="Model training task",
    )
    parser.add_argument("--epochs", default=300, type=int, help="model epochs")
    parser.add_argument("--batch", default=-1, type=int, help="model batch")
    parser.add_argument(
        "-mp", "--modelparams", type=Path, help="model parameters yaml file"
    )
    parser.add_argument(
        "-hp", "--hyperparams", type=Path, help="hyperparameters yaml file"
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args: Namespace = parser.parse_args()
    if not args.model:
        args.model = DEFAULT_MODEL[args.task]

    for arg_key in ("hyperparams", "modelparams"):
        arg_val = getattr(args, arg_key)
        if arg_val and arg_val.exists():
            with arg_val.open() as fp:
                setattr(args, arg_key, yaml.safe_load(fp))
        else:
            setattr(args, arg_key, {})

    freeze_support()

    main(args)
