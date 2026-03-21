import torch
from appmax.applications import mnist
import appmax.evaluation


def test_evaluation():
    model = mnist.SmallDenseNet().eval()
    model_approx = mnist.SmallDenseNet().eval().round(bits=8)
    # batch-like sample
    sample_b = torch.testing.make_tensor((1, 1, 28, 28), dtype=torch.float32, device='cpu', low=0.0, high=2.0)
    eval_net = appmax.evaluation.EvaluationNet(model, model_approx, bounds=(0.0, 2.0)).eval()
    net_err = eval_net(sample_b).item()
    comp_err = appmax.evaluation.compute_error_tensor(model, model_approx, sample_b).item()
    torch.testing.assert_close(net_err, comp_err, atol=1e-5, rtol=0)
