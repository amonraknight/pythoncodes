import torch
import copy


class Rollout(object):

    def __init__(self, model, update_rate, device):
        self.ori_model = model
        self.own_model = copy.deepcopy(model)
        self.update_rate = update_rate
        self.device = device

    # x is the indexes in (batch_size, seq_len)
    # num is the roll-out number
    def get_reward(self, x, num, discriminator, upper_context):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        rewards = torch.zeros(seq_len, batch_size, device=self.device)

        for i in range(num):
            for j in range(1, seq_len):
                data = x[:, 0:j]
                samples = self.own_model.sample(upper_context, batch_size, seq_len, data)
                score = discriminator(samples)
                rewards[j - 1] = rewards[j - 1] + score.squeeze(1)

            score = discriminator(x)
            rewards[seq_len - 1] = rewards[seq_len - 1] + score.squeeze(1)

        rewards = rewards / num
        rewards = rewards.transpose(0, 1).contiguous()
        return rewards

    def update_module(self):
        '''
            dic = {}
            for name, param in self.ori_model.named_parameters():
                dic[name] = param.data
            for name, param in self.own_model.named_parameters():
                # Don't touch the embedding layer.
                if name.startswith('emb'):
                    param.data = dic[name]
                else:
                    # Update gradually the parameters.
                    param.data = self.update_rate * param.data + (1 - self.update_rate) * dic[name]
        '''
        self.own_model.load_state_dict(self.ori_model.state_dict())
