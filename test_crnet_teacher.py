"""No-TTA evaluation for checkpoints produced by train_crnet_teacher.py."""

import test as project_test

from modeling.crnet_teacher import build_crnet_teacher


def _build_without_pretrained_download(args):
    # test.py reconstructs crop_size from --tile_size only after loading the
    # checkpoint args; the checkpoint therefore remains the source of truth.
    return build_crnet_teacher(args, pretrained=False)


project_test.project_train.build_model = _build_without_pretrained_download


if __name__ == "__main__":
    project_test.main()
