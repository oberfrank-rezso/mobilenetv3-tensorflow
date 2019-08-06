# Copyright 2019 Bisonai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of paper Searching for MobileNetV3, https://arxiv.org/abs/1905.02244

Training script
"""
from argparse import ArgumentParser

import tensorflow as tf

from datasets import build_dataset
from mobilenetv3_factory import build_mobilenetv3

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

_available_datasets = [
    "mnist",
    "cifar10",
]

_available_optimizers = {
    "rmsprop": tf.train.RMSPropOptimizer,
    "adam": tf.train.AdamOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
}


def main(args):
    if args.dataset not in _available_datasets:
        raise NotImplementedError

    dataset = build_dataset(
        name=args.dataset,
        shape=(args.height, args.width),
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size
    )

    model = build_mobilenetv3(
        args.model_type,
        input_shape=(args.height, args.width, dataset["channels"]),
        num_classes=dataset["num_classes"],
        width_multiplier=args.width_multiplier,
        l2_reg=args.l2_reg,
    )

    if args.optimizer not in _available_optimizers:
        raise NotImplementedError

    model.compile(
        optimizer=_available_optimizers.get(args.optimizer)(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=args.logdir),
    ]

    model.fit(
        dataset["train"].make_one_shot_iterator(),
        steps_per_epoch=(dataset["num_train"] // args.train_batch_size) + 1,
        epochs=args.num_epoch,
        validation_data=dataset["test"],
        validation_steps=(dataset["num_test"] // args.valid_batch_size) + 1,
        callbacks=callbacks,
    )

    model.save_weights("mobilenetv3_{}_{}_{}.h5".format(args.model_type, args.dataset, args.num_epoch))


if __name__ == "__main__":
    parser = ArgumentParser()

    # Model
    parser.add_argument("--model_type", type=str, default="small", choices=["small", "large"])
    parser.add_argument("--width_multiplier", type=float, default=1.0)

    # Input
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dataset", type=str, default="mnist", choices=_available_datasets)

    # Optimizer
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="rmsprop", choices=_available_optimizers.keys())
    parser.add_argument("--l2_reg", type=float, default=1e-5)

    # Training & validation
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--valid_batch_size", type=int, default=256)
    parser.add_argument("--num_epoch", type=int, default=10)

    # Others
    parser.add_argument("--logdir", type=str, default="logdir")

    args = parser.parse_args()
    main(args)
