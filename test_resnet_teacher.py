"""No-TTA evaluation for checkpoints produced by train_resnet_teacher.py."""

import test as project_test

from modeling.resnet_teacher import build_resnet_teacher


def _build_without_pretrained_download(args):
    return build_resnet_teacher(args, pretrained=False)


project_test.project_train.build_model = _build_without_pretrained_download


if __name__ == "__main__":
    project_test.main()
