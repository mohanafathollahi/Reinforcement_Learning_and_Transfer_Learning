from torch import nn
import torch as th
from utils.penultimate import penultimate

class RewardShaper:
    def __init__(self, embeddings, associated_q_vals, gamma, device):
        self.device = device
        self.embeddings = embeddings.to(self.device)
        self.associated_q_vals = associated_q_vals.to(self.device)
        self.gamma = th.tensor(gamma).to(self.device)
        self.embeddings = nn.functional.normalize(self.embeddings, p=2, dim=1) #normalizing based on dim =1
    def _get_state_action_embedding(self, state, action):
        pass

    def _compute_phi_s_a(self, state, action):
        # state = state.to(self.device)
        # action = action.to(self.device)
        embedding = self._get_state_action_embedding(state, action)
        embedding = nn.functional.normalize(embedding, p=2, dim=0) #normalizing based on dim 0
        similarity_scores = th.matmul(self.embeddings, embedding)
        self.associated_q_vals = self.associated_q_vals.squeeze()
        sum_weighted_q_val_score = th.dot(self.associated_q_vals, similarity_scores)
        avg_weighted_q_val_score = sum_weighted_q_val_score / len(self.associated_q_vals)
        return avg_weighted_q_val_score

    def get_auxiliary_reward(self, state, action, next_state, next_action):
        phi_s_a = self._compute_phi_s_a(state, action)
        next_phi_s_a = self._compute_phi_s_a(next_state, next_action)
        aux_reward = self.gamma * next_phi_s_a - phi_s_a
        return aux_reward

class RewardShaperGeneral(RewardShaper):
    def __init__(self, model, embeddings, associated_q_vals, gamma, device, embedding_type):
        self.model = model
        self.embedding_type = embedding_type
        super(RewardShaperGeneral, self).__init__(embeddings, associated_q_vals, gamma, device)

    def _get_state_action_embedding(self, state, action):
        critic_network = self.model.critic
        policy = self.model.policy
        if isinstance(state, th.Tensor):
            tensor_obs = state.unsqueeze(0).to(self.device)
        else:
            tensor_obs, _ = policy.obs_to_tensor(state)
            tensor_obs = tensor_obs.to(self.device)
        # tensor_act = action.detach().clone().reshape(1, -1).to(self.device)
        tensor_act = th.as_tensor(action, device=self.device).reshape(1, -1)

        policy.set_training_mode(False)
        with th.no_grad():
            q_embedding_net0, q_value_net0, q_embedding_net1, q_value_net1, _, _  = penultimate(tensor_obs, tensor_act, self.model)
            if self.embedding_type == "criticNet0":
                q_embedding = th.squeeze(q_embedding_net0)
            elif self.embedding_type == "criticNet1":
                q_embedding = th.squeeze(q_embedding_net1)
            return q_embedding

#TODO: rewrad shaper for TD3