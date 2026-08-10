import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros(
            (self.num_class, self.num_class),
            dtype=np.float64
        )

    def Pixel_Accuracy(self):
        total = self.confusion_matrix.sum()

        if total == 0:
            return 0.0

        return float(
            np.diag(self.confusion_matrix).sum() / total
        )

    def Pixel_Accuracy_Class(self):
        denominator = self.confusion_matrix.sum(axis=1)

        acc = np.divide(
            np.diag(self.confusion_matrix),
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator != 0
        )

        return float(np.nanmean(acc))

    def Mean_Intersection_over_Union(self):
        intersection = np.diag(self.confusion_matrix)

        union = (
            self.confusion_matrix.sum(axis=1)
            + self.confusion_matrix.sum(axis=0)
            - intersection
        )

        iou = np.divide(
            intersection,
            union,
            out=np.full_like(union, np.nan, dtype=np.float64),
            where=union != 0
        )

        if np.all(np.isnan(iou)):
            return 0.0

        return float(np.nanmean(iou))

    def Intersection_over_Union(self):
        # Binary segmentation: class 1 là road
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        fn = self.confusion_matrix[1, 0]

        denominator = tp + fp + fn

        return (
            float(tp / denominator)
            if denominator > 0
            else 0.0
        )

    def Pixel_Precision(self):
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]

        denominator = tp + fp

        return (
            float(tp / denominator)
            if denominator > 0
            else 0.0
        )

    def Pixel_Recall(self):
        tp = self.confusion_matrix[1, 1]
        fn = self.confusion_matrix[1, 0]

        denominator = tp + fn

        return (
            float(tp / denominator)
            if denominator > 0
            else 0.0
        )

    def Pixel_F1(self):
        precision = self.Pixel_Precision()
        recall = self.Pixel_Recall()

        denominator = precision + recall

        return (
            float(2.0 * precision * recall / denominator)
            if denominator > 0
            else 0.0
        )

    def _generate_matrix(self, gt_image, pre_image):
        gt_image = np.asarray(gt_image, dtype=np.int64)
        pre_image = np.asarray(pre_image, dtype=np.int64)

        mask = (
            (gt_image >= 0)
            & (gt_image < self.num_class)
            & (pre_image >= 0)
            & (pre_image < self.num_class)
        )

        labels = (
            self.num_class * gt_image[mask]
            + pre_image[mask]
        )

        count = np.bincount(
            labels,
            minlength=self.num_class ** 2
        )

        return count.reshape(
            self.num_class,
            self.num_class
        )

    def add_batch(self, gt_image, pre_image):
        if gt_image.shape != pre_image.shape:
            raise ValueError(
                f"GT shape {gt_image.shape} khác "
                f"prediction shape {pre_image.shape}"
            )

        self.confusion_matrix += self._generate_matrix(
            gt_image,
            pre_image
        )

    def reset(self):
        self.confusion_matrix = np.zeros(
            (self.num_class, self.num_class),
            dtype=np.float64
        )
