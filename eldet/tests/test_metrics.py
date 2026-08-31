import numpy as np

from eldet.metrics import (
    ImageDetections,
    ImageGroundTruth,
    box_iou_xyxy,
    ca_localization_recall_at_iou,
    gtba_classification_accuracy,
)


def test_box_iou_identical():
    a = np.array([[0, 0, 10, 10]], dtype=np.float64)
    iou = box_iou_xyxy(a, a)
    assert abs(float(iou[0, 0]) - 1.0) < 1e-5


def test_ca_recall_perfect_match():
    gt = ImageGroundTruth(boxes_xyxy=np.array([[0, 0, 10, 10]]), class_ids=np.array([3]))
    pr = ImageDetections(
        boxes_xyxy=np.array([[0, 0, 10, 10]]),
        scores=np.array([0.9]),
        class_ids=np.array([9]),
    )
    r = ca_localization_recall_at_iou([gt], [pr], iou_threshold=0.5)
    assert abs(r - 1.0) < 1e-5


def test_gtba_tau_gate():
    gt = ImageGroundTruth(boxes_xyxy=np.array([[0, 0, 10, 10]]), class_ids=np.array([1]))
    # IoU tiny -> GTBA should not count this pair for tau=0.1 if overlap is near zero
    pr_far = ImageDetections(
        boxes_xyxy=np.array([[100, 100, 110, 110]]),
        scores=np.array([0.99]),
        class_ids=np.array([1]),
    )
    acc = gtba_classification_accuracy([gt], [pr_far], tau=0.1)
    assert acc == 0.0

    pr_close = ImageDetections(
        boxes_xyxy=np.array([[0, 0, 10, 10]]),
        scores=np.array([0.5]),
        class_ids=np.array([1]),
    )
    acc2 = gtba_classification_accuracy([gt], [pr_close], tau=0.1)
    assert abs(acc2 - 1.0) < 1e-5
