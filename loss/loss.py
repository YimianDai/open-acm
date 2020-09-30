from gluoncv.loss import Loss as gcvLoss


class SoftIoULoss(gcvLoss):
    def __init__(self, batch_axis=0, weight=None):
        super(SoftIoULoss, self).__init__(weight, batch_axis)

    def hybrid_forward(self, F, pred, target):
        # Old One
        pred = F.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() -
                                                intersection.sum() + smooth)
        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss

