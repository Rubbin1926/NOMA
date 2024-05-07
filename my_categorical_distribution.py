import torch


class myCategoricalDistribution:
    def __init__(self, tensor):
        self.logit = tensor
        self.numberOfJobs = tensor.shape[0]
        self.numberOfMachines = tensor.shape[1] - self.numberOfJobs


    # def log_prob(self, value):
    #     value = value.long().unsqueeze(-1)
    #     log_probs = torch.log(self.logit)
    #     log_prob = log_probs.gather(-1, value).squeeze(-1)
    #     return log_prob


    def entropy(self, probs):
        probs_for_entropy = probs
        probs_for_entropy[probs_for_entropy < 0] = 1
        # 计算每个元素的对数概率
        log_probs = torch.log(probs)
        # 乘以概率本身，得到每个元素的负熵
        neg_entropy = probs * log_probs
        # 沿着两个维度求和
        summed_entropy = torch.sum(neg_entropy, dim=(0, 1))
        # 取负数得到熵
        entropy = -summed_entropy
        return entropy


    def action(self):
        return torch.argmax(self.logit.flatten())



if __name__ == "__main__":
    tensor = torch.tensor(
    [[-8.7997e+02, 1.2037e+02, 1.2010e+02, 1.1964e+02, -1.0487e-01, -1.0487e-01],
         [-1.0669e+03, -1.0666e+03, -6.6809e+01, -6.7098e+01, -1.0389e-01, -1.0389e-01],
         [-9.7198e+02, -9.7253e+02, -9.7209e+02, 2.8662e+01, -1.0575e-01, -1.0575e-01],
         [-9.6986e+02, -9.6983e+02, -9.6986e+02, -9.6990e+02, -1.0216e-01, -1.0216e-01]], device='cuda:0')
    cd = myCategoricalDistribution(tensor)


    print(cd.action())
    print(cd.entropy(tensor))
