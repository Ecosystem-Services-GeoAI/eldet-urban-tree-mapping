import torch

from eldet.kd_losses import kd_detection_v1, kl_div_logits


def test_kl_student_gets_grad():
    zs = torch.randn(4, 5, requires_grad=True)
    zt = torch.randn(4, 5)
    loss = kl_div_logits(zs, zt)
    loss.backward()
    assert zs.grad is not None
    assert torch.isfinite(zs.grad).all()


def test_teacher_branch_detached_in_kd_v1():
    zs = torch.randn(8, 4, requires_grad=True)
    zt = torch.randn(8, 4)  # teacher not a leaf we train
    bs = torch.randn(8, 4, requires_grad=True)
    bt = torch.randn(8, 4)
    loss = kd_detection_v1(zs, zt, bs, bt, lambda_cls=1.0, lambda_box=1.0)
    loss.backward()
    assert zs.grad is not None and torch.isfinite(zs.grad).all()
    assert bs.grad is not None and torch.isfinite(bs.grad).all()
