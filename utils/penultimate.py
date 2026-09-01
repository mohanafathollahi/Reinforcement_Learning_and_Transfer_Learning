import torch
from torch import Tensor, nn
#the tensor obs and tensor act should be normalized and passed to the penultimate function
def penultimate(tensor_obs: Tensor,
                tensor_act: Tensor,
                model: nn.Module) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    qf0 = model.policy.critic.qf0
    qf1 = model.policy.critic.qf1

    with torch.no_grad():
        x = torch.cat([tensor_obs, tensor_act], dim=1) #when the tensor have batch size
        penultimate_qf0 = qf0[:-1](x)
        q_value_qf0 = qf0[-1](penultimate_qf0)

        penultimate_qf1 = qf1[:-1](x)
        q_value_qf1 = qf1[-1](penultimate_qf1)

    pen_min_qf0_qf1 = torch.minimum(penultimate_qf0, penultimate_qf1)
    q_value_min_qf0_qf1 = torch.minimum(q_value_qf0, q_value_qf1)

    return penultimate_qf0, q_value_qf0, penultimate_qf1, q_value_qf1, pen_min_qf0_qf1, q_value_min_qf0_qf1