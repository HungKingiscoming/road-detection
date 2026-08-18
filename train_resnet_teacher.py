"""Train the independent ResNet road teacher with the existing pipeline.

Copy this file beside the project's train.py and copy resnet_teacher.py under
modeling/.  Teacher-only command-line options are removed before delegating to
the existing parser, then added back to the saved Namespace/checkpoint.
"""

from __future__ import annotations

import argparse
import sys

import train as project_train

from modeling.resnet_teacher import build_resnet_teacher


_original_parse_args = project_train.parse_args


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--teacher_arch", choices=("resnet34", "resnet50"), default="resnet50"
    )
    parser.add_argument("--teacher_decoder_channels", type=int, default=256)
    parser.add_argument(
        "--teacher_pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--teacher_freeze_backbone_bn",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    custom, remaining = parser.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        args = _original_parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(custom).items():
        setattr(args, key, value)
    args.model_role = "resnet_teacher"
    return args


project_train.parse_args = _parse_args
project_train.build_model = build_resnet_teacher


if __name__ == "__main__":
    project_train.main()
