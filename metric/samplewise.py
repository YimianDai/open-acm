"""Evaluation Metrics for Semantic Segmentation of Foreground Only"""
import threading
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.metric import EvalMetric

# __all__ = ['SamplewiseSigmoidMetric', 'batch_pix_accuracy', 'batch_intersection_union']
__all__ = ['SamplewiseSigmoidMetric', 'batch_intersection_union']

class SamplewiseSigmoidMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, score_thresh=0.5):
        super(SamplewiseSigmoidMetric, self).__init__('pixAcc & mIoU')
        self.nclass = nclass
        self.score_thresh = score_thresh
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
            inter_arr, union_arr = batch_intersection_union(
                pred, label, self.nclass, self.score_thresh)
            with self.lock:
                self.total_inter = np.append(self.total_inter, inter_arr)
                self.total_union = np.append(self.total_union, union_arr)

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
        # pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return mIoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])


def batch_intersection_union(output, target, nclass, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (nd.sigmoid(output).asnumpy() > score_thresh).astype('int64') # P
    # predict = (output.asnumpy() > 0).astype('int64') # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64') # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target) # TP


    num_sample = intersection.shape[0]
    area_inter_arr = np.zeros(num_sample)
    area_pred_arr = np.zeros(num_sample)
    area_lab_arr = np.zeros(num_sample)
    area_union_arr = np.zeros(num_sample)
    for b in range(num_sample):

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
        area_inter_arr[b] = area_inter

        area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
        area_pred_arr[b] = area_pred

        area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
        area_lab_arr[b] = area_lab

        area_union = area_pred + area_lab - area_inter
        area_union_arr[b] = area_union

        assert (area_inter <= area_union).all(), \
            "Intersection area should be smaller than Union area"

    return area_inter_arr, area_union_arr



class ROCMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):
        super(ROCMetric, self).__init__('ROC')
        self.nclass = nclass
        self.lock = threading.Lock()
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        # self.reset()

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
            for iBin in range(self.bins+1):
                score_thresh = (iBin + 0.0) / self.bins
                # print(iBin, "-th, score_thresh: ", score_thresh)
                i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(pred, label, self.nclass,
                                                             score_thresh)
                # print("i_tp: ", i_tp)
                # print("i_fp: ", i_fp)
                with self.lock:
                    self.tp_arr[iBin] += i_tp
                    self.pos_arr[iBin] += i_pos
                    self.fp_arr[iBin] += i_fp
                    self.neg_arr[iBin] += i_neg

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
        # pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates
        # return self.tp_arr, self.fp_arr

    # def reset(self):
    #     """Resets the internal evaluation result to initial state."""
    #     self.tp_arr = np.ones(self.bins+1)
    #     self.pos_arr = np.ones(self.bins+1)
    #     self.fp_arr = np.ones(self.bins+1)
    #     self.neg_arr = np.ones(self.bins+1)




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (nd.sigmoid(output).asnumpy() > score_thresh).astype('int64') # P
    # predict = (output.asnumpy() > 0).astype('int64')  # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn

    return tp, pos, fp, neg


def cal_normalized_tp_pos_fp_neg(output, target, nclass, score_thresh):
    """mIoU"""
    # inputs are NDarray, output 4D, target 3D
    # the category 0 is ignored class, typically for background / boundary
    mini = 1
    maxi = 1 # nclass
    nbins = 1 # nclass

    predict = (nd.sigmoid(output).asnumpy() > score_thresh).astype('int64') # P
    # predict = (output.asnumpy() > 0).astype('int64')  # P
    if len(target.shape) == 3:
        target = nd.expand_dims(target, axis=1).asnumpy().astype('int64') # T
    elif len(target.shape) == 4:
        target = target.asnumpy().astype('int64')  # T
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * (predict == target)  # TP
    tp = intersection.sum()
    fp = (predict * (predict != target)).sum()  # FP
    tn = ((1 - predict) * (predict == target)).sum()  # TN
    fn = ((predict != target) * (1 - predict)).sum()   # FN
    pos = tp + fn
    neg = fp + tn

    return tp, pos, fp, neg



class nROCMetric(EvalMetric):
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):
        super(nROCMetric, self).__init__('ROC')
        self.nclass = nclass
        self.lock = threading.Lock()
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        # self.reset()

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
            for iBin in range(self.bins+1):
                score_thresh = (iBin + 0.0) / self.bins
                # print(iBin, "-th, score_thresh: ", score_thresh)
                i_tp, i_pos, i_fp, i_neg = cal_tp_pos_fp_neg(pred, label, self.nclass,
                                                             score_thresh)
                # print("i_tp: ", i_tp)
                # print("i_fp: ", i_fp)
                with self.lock:
                    self.tp_arr[iBin] += i_tp
                    self.pos_arr[iBin] += i_pos
                    self.fp_arr[iBin] += i_fp
                    self.neg_arr[iBin] += i_neg

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
        tp_rates = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates = self.fp_arr / (self.neg_arr + 0.001)

        return tp_rates, fp_rates
