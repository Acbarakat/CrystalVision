# -*- coding: utf-8 -*-
"""
Create frozen graphs to be used by OpenCV

Todo:
    * Find more varied data, such as off-center or card with background

"""
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from testmodels import CATEGORIES, DATA_DIR


def freeze_model(model_path, model_name, frozen_path=None):
    '''
    Create and save model and catergory data

    Args:
        model_path (str): The directory of models
        model_name (str): The name (dir) of the saved model
        frozen_path: Where the frozen models will save to
            (default is DATA_DIR\\frozen)
    '''
    if frozen_path is None:
        frozen_path = os.path.join(DATA_DIR, "frozen")

    model = keras.models.load_model(os.path.join(model_path, model_name))

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)

    print(model.name)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(frozen_func.graph, logdir=frozen_path, name=f"{model_name}.pb", as_text=False)

    pb_file = os.path.join(frozen_path, f"{model_name}.pb")
    graph_def = tf.compat.v1.GraphDef()

    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

    for i in reversed(range(len(graph_def.node))):
        if graph_def.node[i].op in ('Const', 'NoOp'):
            del graph_def.node[i]
    graph_def.library.Clear()

    tf.compat.v1.train.write_graph(graph_def, logdir=frozen_path, name=f"{model_name}.pbtxt", as_text=True)


def main() -> None:
    for category in CATEGORIES:
        model_path = os.path.join(DATA_DIR, "model")
        freeze_model(model_path, f"{category}_model")


if __name__ == '__main__':
	main()