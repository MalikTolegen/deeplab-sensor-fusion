import numpy as np


def meanIoU(preds, targets):
    if preds.shape != targets.shape:
        print("preds dimension: ", preds.shape, ", targets dimension:", targets.shape)
        return
        
    # if preds.dim() != 4:
    #     print("target has dim", preds.dim(), ", Must be 4.")
    #     return
    
    iousum = 0
    for i in range(preds.shape[0]):
        target_arr = preds[i, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = targets[i, :, :].clone().detach().cpu().numpy().argmax(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else :
            iou_score = intersection / union
        iousum +=iou_score

    miou = iousum/preds.shape[0]

    return miou