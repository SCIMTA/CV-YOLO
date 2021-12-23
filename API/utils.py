import numpy as np


# def iou_cal(boxA, boxB):
#     # determine the (x, y)-coordinates of the intersection rectangle
#     xA = max(boxA[0], boxB[0])
#     yA = max(boxA[1], boxB[1])
#     xB = min(-boxA[2] + boxA[0], -boxB[2] + boxB[0])
#     yB = min(-boxA[3] + boxA[1], -boxB[3] + boxB[1])
#
#     # compute the area of intersection rectangle
#     interArea = (xB - xA) * (yB - yA)
#
#     # compute the area of both the prediction and ground-truth
#     # rectangles
#     boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
#     boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
#
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = interArea / float(boxAArea + boxBArea - interArea)
#
#     # return the intersection over union value
#     return iou

def iou_cal(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[2])
    the  right-up coordinate of  pred_box:(pred_box[1], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[1], gt_box[1])
    iymin = max(pred_box[2], gt_box[2])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = ((pred_box[1] - pred_box[0]) * (pred_box[3] - pred_box[2]) +
           (gt_box[1] - gt_box[0]) * (gt_box[3] - gt_box[2]) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def check_iou(list_box):
    attemp = -1
    new_list_box = []
    # for box1 in list_box:
    while len(list_box) > 0:
        attemp += 1
        box1 = list_box[0]
        list_overlap = [box1]
        for box2 in list_box:
            if not check_same_box(box1, box2):
                if iou_cal(box1, box2) > 0.7:
                    list_overlap.append(box2)
        # print(list_overlap)
        max_confidence = get_max_confidence(list_overlap)
        new_list_box.append(max_confidence)
        # print('new list overlap')
        # print(f'max_confidence = {max_confidence}')

        list_box = [box for box in list_box if not check_include(box, list_overlap)]

        # for box2 in list_overlap:
        #     # if not check_same_box(box1, box2):
        #     if check_include(box2, list_box):
        #         list_box.remove(box2)
        #         print(box2)
        # print(len(list_box))
    # print(len(new_list_box))
    # print(f'len(list_box) = {type(list_box)}')
    # print(new_list_box)
    return new_list_box


def check_include(check_box, list_box):
    for box in list_box:
        if check_same_box(check_box, box):
            return True
    return False


def get_max_confidence(list_box):
    max_confidence = list_box[0]

    for idx, box in enumerate(list_box):
        if box[4] > max_confidence[4]:
            max_confidence = box
    return max_confidence


def check_same_box(box1, box2):
    if box1[0] == box2[0] and box1[1] == box2[1] and box1[2] == box2[2] and box1[3] == box2[3]:
        return True
    return False
