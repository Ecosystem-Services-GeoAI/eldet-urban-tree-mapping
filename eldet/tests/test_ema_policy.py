import torch
from torch import nn

from eldet.ema_policy import update_teacher_from_student, yolo_detect_cls_param


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_w = nn.Parameter(torch.ones(2))
        self.head_cv3_w = nn.Parameter(torch.zeros(2))


def test_ema_updates_cls_faster():
    teacher = Toy()
    student = Toy()
    teacher.backbone_w.data.zero_()
    teacher.head_cv3_w.data.zero_()
    student.backbone_w.data.fill_(10.0)
    student.head_cv3_w.data.fill_(5.0)

    update_teacher_from_student(
        teacher,
        student,
        global_decay=0.9,
        cls_decay=0.1,
        is_cls_param=yolo_detect_cls_param,
        ema_buffers=False,
    )
    # backbone: 0*0.9 + 10*0.1 = 1
    assert torch.allclose(teacher.backbone_w, torch.tensor([1.0, 1.0]))
    # cv3: 0*0.1 + 5*0.9 = 4.5
    assert torch.allclose(teacher.head_cv3_w, torch.tensor([4.5, 4.5]))


def test_ema_global_only():
    teacher = Toy()
    student = Toy()
    teacher.backbone_w.data.zero_()
    teacher.head_cv3_w.data.zero_()
    student.backbone_w.data.fill_(2.0)
    student.head_cv3_w.data.fill_(2.0)
    update_teacher_from_student(teacher, student, global_decay=0.5, ema_buffers=False)
    assert torch.allclose(teacher.backbone_w, torch.ones(2))
    assert torch.allclose(teacher.head_cv3_w, torch.ones(2))
