# -*- coding: utf-8 -*-
"""
Generate various models.

Todo:
    * Find more varied data, such as off-center or card with background

"""
import os
import json
from typing import Any, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split
from keras.utils import image_utils
from keras.utils.image_dataset import paths_and_labels_to_dataset

from data.gather import DATA_DIR
from data.dataset import make_database, extendDataset


def make_model(train_ds: tf.data.Dataset,
               validation_ds: tf.data.Dataset,
               image_shape: tuple,
               label_count: int,
               model_type: str = "custom",
               epochs: int = 100,
               seed: Any = None,
               model_name: str = "fftcg_model") -> None:
    """
    Create and save model.

    Args:
        train_ds (tf.data.Dataset): Training data
        validation_ds (tf.data.Dataset): Test data
        image_shape (tuple): The (height, width) of the image data
        label_count (int): The number of `labels`
        model_type (str): String describing the type of `model`
            (default is `custom`)
        epochs (int): The maximum number of training iterations
            (default is `30`)
        seed (Any): Optional random seed for shuffling, transformations,
            and dropouts.
        model_name (str): The name of the model
            (default is `fftcg_model`)
    """
    tf.keras.backend.clear_session()

    loss = tf.keras.losses.CategoricalCrossentropy()
    metrics = [tf.keras.metrics.CategoricalAccuracy(name='card_accuracy')]

    if model_type == "custom":
        # TODO: A well crafted custom per attribute is required
        model = models.Sequential(name="custom", layers=[
            layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            layers.Dense(2 ** 6, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        # optimizers = [tf.keras.optimizers.RMSprop()]
        optimizers = [tf.keras.optimizers.SGD(learning_rate=0.015, momentum=0.9, nesterov=True)]
    elif model_type == "name_en":
        model = models.Sequential(name="name_en", layers=[
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            layers.Dense(2 ** 6, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizers = [tf.keras.optimizers.RMSprop()]
    elif model_type in ("burst", "multicard", "mono"):
        model = models.Sequential(name=model_type, layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            layers.Dense(2 ** 6, activation='relu'),
            layers.Dense(1, activation="sigmoid")
        ])
        optimizers = [
            # tf.keras.optimizers.RMSprop(centered=True),
            #tf.keras.optimizers.RMSprop(centered=False),
            # tf.keras.optimizers.Adam(amsgrad=True),
            tf.keras.optimizers.Adam(amsgrad=False),
            # tf.keras.optimizers.Nadam(),
        ]
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [tf.keras.metrics.BinaryAccuracy(name='card_accuracy')]
    elif model_type == "cost":
        model = models.Sequential(name="cost", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            # layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 5, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizers = [
            tf.keras.optimizers.RMSprop(centered=True),
            tf.keras.optimizers.RMSprop(centered=False),
            tf.keras.optimizers.Adam(amsgrad=True),
            tf.keras.optimizers.Adam(amsgrad=False),
            tf.keras.optimizers.Nadam(),
            tf.keras.optimizers.SGD(learning_rate=0.01,
                                    momentum=0.9,
                                    nesterov=True),
        ]
    elif model_type == "power":
        model = models.Sequential(name="power", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Dense(2 ** 6, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizers = [tf.keras.optimizers.RMSprop()]
    elif model_type == "element":
        model = models.Sequential(name="element", layers=[
            layers.Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(padding='same'),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizers = [
            # tf.keras.optimizers.RMSprop(centered=True),
            # tf.keras.optimizers.RMSprop(centered=False),
            tf.keras.optimizers.Adam(amsgrad=True),
            # tf.keras.optimizers.Adam(amsgrad=False),
            # tf.keras.optimizers.Nadam(),
            # tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9, nesterov=True),
            # tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.9),
        ]
    elif model_type == "type_en":
        model = models.Sequential(name="type_en", layers=[
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=image_shape),
            layers.MaxPooling2D(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(2 ** 8, activation='relu'),
            # layers.Dropout(0.2, seed=seed),
            layers.Dense(label_count, activation="softmax")
        ])
        optimizers = [tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)]
    else:
        raise RuntimeError(f"model_type '{model_type}' is not supported")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_card_accuracy',
                                         min_delta=0.005,
                                         patience=2,
                                         restore_best_weights=True),
        #tf.keras.callbacks.EarlyStopping(monitor='card_accuracy',
        #                                 min_delta=0.005,
        #                                 patience=2,
        #                                 restore_best_weights=True),
    ]

    base_name = model.name
    for idx, optimizer in enumerate(optimizers):
        m = models.clone_model(model)
        m.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
        # print(model.summary())
        m._name = f"{base_name}_{idx + 1}"
        print(m.name)

        m.fit(train_ds,
              epochs=epochs,
              validation_data=validation_ds,
              steps_per_epoch=len(train_ds),  # // train_ds._batch_size,
              validation_steps=len(validation_ds),  # // validation_ds._batch_size,
              callbacks=callbacks)

        m.save(os.path.join(DATA_DIR,
                            "model",
                            f"{model_name}_{idx + 1}.h5"))


def generate(df: pd.DataFrame,
             key: str,
             stratify: list,
             model_type: str,
             image_size: Tuple[int, int] = (250, 179),
             label_mode: str = 'categorical',
             batch_size: int = 64,
             shuffle: bool = True,
             seed: Any = None,
             interpolation: str = "bilinear",
             crop_to_aspect_ratio: bool = False) -> None:
    """
    Create and save model and catergory data.

    Args:
        df (pd.DataFrame): All card information
        image_key (str): The (height, width) of the image data
        stratify (list): The columns by which we stratify training and test datasets
        model_type (str): String describing the type of `model`
            (default is `resnet`)
        image_shape (tuple): The (height, width) of the image data
            (default is (250, 179))
        label_mode (str): String describing the encoding of `labels`. Options are:
          - 'int': means that the labels are encoded as integers
              (e.g. for `sparse_categorical_crossentropy` loss).
          - 'categorical' means that the labels are
              encoded as a categorical vector
              (e.g. for `categorical_crossentropy` loss).
          - 'binary' means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0 or 1
              (e.g. for `binary_crossentropy`).
          - None (no labels).
        batch_size (int): Size of the batches of data.
            If `None`, the data will not be batched
            (default is 32)
        shuffle (bool): Whether to shuffle the data.
            If set to False, sorts the data in alphanumeric order.
            (default is True)
        seed (Any): Optional random seed for shuffling and transformations.
        interpolation (str): The interpolation method used when resizing images.
            Supports `bilinear`, `nearest`, `bicubic`, `area`,
            `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
            (default is `bilinear`)
        crop_to_aspect_ratio: If True, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the target
            aspect ratio, the output image will be cropped so as to return the
            largest possible window in the image (of size `image_size`) that matches
            the target aspect ratio.
            (default is False)
    """
    codes, uniques = df[key].factorize()

    X_train, X_test, y_train, y_test = train_test_split(df['filename'],
                                                        codes,
                                                        test_size=0.33,
                                                        random_state=seed,
                                                        shuffle=shuffle,
                                                        stratify=df[stratify])

    training_dataset = paths_and_labels_to_dataset(
        image_paths=X_train.tolist(),
        image_size=image_size,
        num_channels=3,
        labels=y_train.tolist(),
        label_mode=label_mode,
        num_classes=len(uniques),
        interpolation=interpolation,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )

    testing_dataset = paths_and_labels_to_dataset(
        image_paths=X_test.tolist(),
        image_size=image_size,
        num_channels=3,
        labels=y_test.tolist(),
        label_mode=label_mode,
        num_classes=len(uniques),
        interpolation=interpolation,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )

    training_dataset = extendDataset(training_dataset, seed, "training",
                                     batch_size=batch_size)

    testing_dataset = extendDataset(testing_dataset, seed, "test",
                                    batch_size=batch_size)

    MODEL_PATH = os.path.join(DATA_DIR, "model")
    if not os.path.exists(MODEL_PATH + os.sep):
        os.makedirs(MODEL_PATH + os.sep)

    with open(os.path.join(MODEL_PATH, f"{key}.json"), "w+") as fp:
        json.dump(uniques.to_list(), fp)

    make_model(training_dataset,
               testing_dataset,
               seed=seed,
               model_type=model_type,
               model_name=f"{key}_model",
               image_shape=(image_size[0], image_size[1], 3),
               label_count=len(uniques))


def main(image: str = "thumbs",
         seed: Any = None,
         interpolation: str = "bilinear") -> None:
    """
    Create and save all models based on card data.

    Args:
        image (str): The column to explode containing lists of image sources.
            (default is `thumbs`)
        seed (Any): Optional random seed for shuffling and transformations.
            (default is None)
        interpolation (str): The interpolation method used when resizing images.
            Supports `bilinear`, `nearest`, `bicubic`, `area`,
            `lanczos3`, `lanczos5`, `gaussian`, `mitchellcubic`.
            (default is `bilinear`)
    """
    if seed is None:
        seed = np.random.randint(1e6)
    interpolation = image_utils.get_interpolation(interpolation)

    df = make_database().explode(image)
    # df["Class"] = df['thumbs'].apply(lambda x: 'Full Art' if '_fl' in x.lower() else '')

    # Ignore Crystal Tokens
    df.query("type_en != 'Crystal'", inplace=True)

    # Ignore Boss Deck cards
    df.query("rarity != 'B'", inplace=True)

    # Ignore Full Art Cards
    df.query(f"~{image}.str.contains('_FL') and ~{image}.str.contains('_2_')",
             inplace=True)

    # Ignore Promo Cards, they tend to be Full Art
    df.query(f"~{image}.str.contains('_PR')", inplace=True)

    # Ignore
    df.query(f"~{image}.str.contains('_premium')", inplace=True)

    # Ignore by language
    # df = df.query(f"~{image}.str.contains('_eg')")  # English
    # df = df.query(f"~{image}.str.contains('_fr')")  # French
    # df = df.query(f"~{image}.str.contains('_es')")  # Spanish
    # df = df.query(f"~{image}.str.contains('_it')")  # Italian
    # df = df.query(f"~{image}.str.contains('_de')")  # German
    # df = df.query(f"~{image}.str.contains('_jp')")  # Japanese

    # WA: Bad Download/Image from server
    df.query(f"{image} not in ('8-080C_es.jpg', '11-138S_fr.jpg', '12-049H_fr_Premium.jpg', '13-106H_de.jpg')", inplace=True)

    # Source image folder
    df = df.copy()  # WA: for pandas modification on slice
    if image == "images":
        image_dir = os.path.abspath(os.path.join(DATA_DIR, "img"))
        df.rename({"img": "filename"}, axis=1, inplace=True)
    else:
        image_dir = os.path.abspath(os.path.join(DATA_DIR, "thumb"))
        df.rename({"thumbs": "filename"}, axis=1, inplace=True)
    df["filename"] = image_dir + os.sep + df["filename"]
    # df[df["Multicard"] == "\u25cb"]["Name_EN"] = "Generic"
    # df[df["Job_EN"] == "Standard Unit"]["Name_EN"] = "Generic"
    # df.query('multicard != True and job_en != "Standard Unit"', inplace=True)

    model_mapping = (
        # ("name_en", ["name_en", "element", "type_en"], "name_en", "categorical"),
        # ("element", ["element", "type_en"], "element", "categorical"),
        # ("type_en", ["type_en", "element"], "type_en", "categorical"),
        # ("cost", ["cost", "element"], "cost", "categorical"),
        # ("power", ["power", "type_en", "element"], "power", "categorical"),
        ("ex_burst", ["ex_burst", "element", "type_en"], "burst", "binary"),
        ("multicard", ["multicard", "element", "type_en"], "multicard", "binary"),
        ("mono", ["mono", "type_en"], "mono", "binary"),
    )

    for key, stratify, model_type, label_mode in model_mapping:
        generate(df,
                 key,
                 stratify,
                 model_type,
                 label_mode=label_mode,
                 seed=seed,
                 interpolation=interpolation)


if __name__ == '__main__':
    with tf.device('/GPU:0'):
        main(seed=23)