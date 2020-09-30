"""Evaluation Metrics for Semantic Segmentation of Foreground Only"""
import threading
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.metric import EvalMetric

__all__ = ['SigmoidMetric', 'batch_pix_accuracy', 'batch_intersection_union']

class SigmoidMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass):
        super(SigmoidMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.lock = threading.Lock()
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.

        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """
        def evaluate_worker(self, label, pred):
            correct, labeled = batch_pix_accuracy(
                pred, label)
            inter, union = batch_intersection_union(
                pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union

        if isinstance(preds, mx.nd.NDArray):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker,
                                        args=(self, label, pred),
                                       )
                       for (label, pred) in zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        # print("self.total_correct: ", self.total_correct)
        # print("self.total_label: ", self.total_label)
        # print("self.total_union: ", self.total_union)
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    # predict = np.argmax(output.asnumpy(), 1).astype('int64')
    # print("Metric output.shape: ", output.shape)
    # print("Metric target.shape: ", target.shape)
    # print("output.max(): ", output.max().asscalar())
    # print("target.max(): ", target.max().asscalar())
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64') # T
    else:
        raise ValueError("Unknown target dimension")
    # print("output.shape: ", output.shape)
    # print("target.shape: ", target.shape)
    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output.asnumpy() > 0).astype('int64') # P
    pixel_labeled = np.sum(target > 0) # T
    pixel_correct = np.sum((predict == target)*(target > 0)) # TP

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass
    predict = (output.asnumpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64') # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target) # TP
    
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union



