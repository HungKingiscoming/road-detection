"""Train the corrected CRNet teacher through the existing train.py pipeline."""

from __future__ import annotations

import argparse
import sys

import train as project_train

from modeling.crnet_teacher import build_crnet_teacher


_original_parse_args = project_train.parse_args


def _parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--crnet_pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--crnet_freeze_backbone_bn",
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
    args.model_role = "crnet_teacher"
    return args


project_train.parse_args = _parse_args
project_train.build_model = build_crnet_teacher


if __name__ == "__main__":
    project_train.main()
