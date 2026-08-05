import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.precision = 0.0
        self.recall = 0.0

    def Pixel_Accuracy(self):
        total = self.confusion_matrix.sum()
        if total == 0:
            return 0.0
        Acc = np.diag(self.confusion_matrix).sum() / total
        return Acc

    def Pixel_Accuracy_Class(self):
        row_sum = self.confusion_matrix.sum(axis=1)
        acc_per_class = np.divide(
            np.diag(self.confusion_matrix),
            row_sum,
            out=np.zeros_like(row_sum, dtype=float),
            where=row_sum != 0
        )
        return np.nanmean(acc_per_class)

    def Pixel_Precision(self):
        denom = self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1]
        if denom == 0:
            self.precision = 0.0
        else:
            self.precision = self.confusion_matrix[1, 1] / denom
        return self.precision

    def Pixel_Recall(self):
        denom = self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0]
        if denom == 0:
            self.recall = 0.0
        else:
            self.recall = self.confusion_matrix[1, 1] / denom
        return self.recall

    def Pixel_F1(self):
        prec = self.Pixel_Precision()
        rec = self.Pixel_Recall()
        if (prec + rec) == 0:
            return 0.0
        f1 = 2 * prec * rec / (prec + rec)
        return f1

    def Intersection_over_Union(self):
        denom = self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[0, 1]
        if denom == 0:
            return 0.0
        IoU = self.confusion_matrix[1, 1] / (denom + 1e-10)
        return IoU

    def Mean_Intersection_over_Union(self):
        denom = (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        iou = np.divide(
            np.diag(self.confusion_matrix),
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0
        )
        return np.nanmean(iou)

    def Frequency_Weighted_Intersection_over_Union(self):
        total = np.sum(self.confusion_matrix)
        if total == 0:
            return 0.0
        freq = np.sum(self.confusion_matrix, axis=1) / total
        denom = (
            np.sum(self.confusion_matrix, axis=1) +
            np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix)
        )
        iu = np.divide(
            np.diag(self.confusion_matrix),
            denom,
            out=np.zeros_like(denom, dtype=float),
            where=denom != 0
        )
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        # Tự động loại bỏ chiều channel 1 nếu truyền vào shape [B, 1, H, W]
        if gt_image.ndim == 4 and gt_image.shape[1] == 1:
            gt_image = np.squeeze(gt_image, axis=1)
        if pre_image.ndim == 4 and pre_image.shape[1] == 1:
            pre_image = np.squeeze(pre_image, axis=1)

        assert gt_image.shape == pre_image.shape, (
            f"Shape mismatch trong Evaluator: GT {gt_image.shape} vs Pred {pre_image.shape}"
        )
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.precision = 0.0
        self.recall = 0.0
