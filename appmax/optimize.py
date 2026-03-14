import torch
import scipy.optimize
import appmax.neurons


def optimize(message: appmax.neurons.Message, constraints: appmax.neurons.Constraints, bounds: tuple):
    TOL = 0  # 1e-8

    # (U)  Ax + b >= 0
    #         -Ax <= b
    U_weight = -torch.cat(constraints.U_weight)
    U_bias = torch.cat(constraints.U_bias) + TOL

    # (S)  Ax + b <= 0
    #          Ax <= -b
    S_weight = torch.cat(constraints.S_weight)
    S_bias = -torch.cat(constraints.S_bias) + TOL

    c = message.s_weight.squeeze().cpu().numpy()
    A_ub = torch.cat((U_weight, S_weight)).cpu().numpy()
    b_ub = torch.cat((U_bias, S_bias)).cpu().numpy()
    print('starting optimization')
    result = scipy.optimize.linprog(c, A_ub, b_ub, bounds=bounds, options={"disp": True})
    print(result)
