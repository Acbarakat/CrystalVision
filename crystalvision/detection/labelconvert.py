import os
import yaml
import json
from pathlib import Path

import numpy as np
from labelme2yolo.l2y import Labelme2YOLO, extend_point_list, logger as log


class MyLabelme2YOLO(Labelme2YOLO):
    def __init__(self, json_dir, output_format, label_list=None, expand_flags=True):
        self._json_dir = Path(json_dir)
        self._output_format = output_format
        self._label_list = set()
        self._label_id_map = {}
        self._label_dir_path = ""
        self._image_dir_path = ""
        self.expand_flags = expand_flags

        if label_list:
            self._label_list = label_list
            self._label_id_map = {
                label: label_id for label_id, label in enumerate(label_list)
            }
        else:
            # get label list from json files for parallel processing
            for json_file in self._json_dir.glob("**/*.json"):
                with open(json_file, encoding="utf-8") as file:
                    json_data = json.load(file)
                    for shape in json_data["shapes"]:
                        label = shape["label"]
                        self._label_list.add(label)

                        if expand_flags and shape["flags"]:
                            for key, val in shape["flags"].items():
                                if val:
                                    label += f"_{key}"
                                    self._label_list.add(label)

            self._label_list = list(self._label_list)
            self._label_id_map = {
                label: label_id for label_id, label in enumerate(self._label_list)
            }

    def _get_other_shape_yolo_object(self, shape, img_h, img_w):
        point_list = shape["points"]
        points = np.zeros(2 * len(point_list))
        points[::2] = [float(point[0]) / img_w for point in point_list]
        points[1::2] = [float(point[1]) / img_h for point in point_list]

        if self._output_format == "obb" and len(point_list) != 4:
            log.error("Shape has too many points: %s", shape)
            return None

        if len(points) == 4:
            if self._output_format == "polygon":
                points = extend_point_list(points)
            elif self._output_format == "bbox":
                points = extend_point_list(points, "bbox")

        if shape["label"]:
            label = shape["label"]
            if self.expand_flags and shape["flags"]:
                for key, val in shape["flags"].items():
                    if val:
                        label += f"_{key}"
            self._update_id_map(label)
            label_id = self._label_id_map[label]

            return label_id, points.tolist()

        return None

    def _save_dataset_yaml(self):
        yaml_path = os.path.join(self._json_dir, "YOLODataset/", "dataset.yaml")

        with open(yaml_path, "w+", encoding="utf-8") as yaml_file:
            data = {
                "path": os.path.abspath(self._json_dir),
                "train": "./YOLODataset/images/train/",
                "val": "./YOLODataset/images/val/",
                "test": "./YOLODataset/images/test/",
                "nc": len(self._label_id_map),
                "names": list(self._label_id_map.keys()),
            }

            test_dir = os.path.join(self._image_dir_path, "test/")
            if not os.path.exists(test_dir):
                del data["test"]

            yaml.dump(data, yaml_file)


if __name__ == "__main__":
    import argparse
    from labelme2yolo.__about__ import __version__

    parser = argparse.ArgumentParser("labelme2yolo")
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + __version__ + "-custom",
    )
    parser.add_argument(
        "--json_dir",
        required=True,
        type=str,
        help="Please input the path of the labelme json files.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        nargs="?",
        default=0.2,
        help="Please input the validation dataset size, for example 0.2.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        nargs="?",
        default=0.0,
        help="Please input the test dataset size, for example 0.1.",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        default="polygon",
        help='The default output format for labelme2yolo is "polygon".'
        ' However, you can choose to output in bbox format by specifying the "bbox" option.',
    )
    parser.add_argument(
        "--label_list", type=set, nargs="+", help="A list or set of labels"
    )
    parser.add_argument(
        "-f", "--flags", action="store_true", help="Expand labels by their flags."
    )

    args = parser.parse_args()

    convertor = MyLabelme2YOLO(
        args.json_dir, args.output_format, args.label_list, args.flags
    )

    convertor.convert(val_size=args.val_size, test_size=args.test_size)
