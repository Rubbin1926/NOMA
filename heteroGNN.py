import dgl
import torch
from torch import nn
from torch.nn import functional as F
import dgl.nn.pytorch as dglnn


def tensor_to_dgl(Graph: torch.tensor):
    numberOfJobs, numberOfMachines = Graph.shape[0], Graph.shape[1] - Graph.shape[0]
    job_to_job, job_to_machine = Graph[:, :numberOfJobs], Graph[:, numberOfJobs:]

    def tensor_to_link(mini_Graph):
        nonzero_indices = torch.nonzero(mini_Graph)
        nonzero_indices_2d = nonzero_indices.t()
        re = nonzero_indices_2d.view(2, -1)
        return re

    job_to_job_link_src, job_to_job_link_dst = tensor_to_link(job_to_job)[0], tensor_to_link(job_to_job)[1]
    job_to_machine_link_src, job_to_machine_link_dst = tensor_to_link(job_to_machine)[0], tensor_to_link(job_to_machine)[1]

    g = dgl.heterograph({('job', 'to_job', 'job'): ([], []),
                         ('job', 'to_machine', 'machine'): ([], [])})

    g.add_edges(job_to_job_link_src, job_to_job_link_dst, etype=('job', 'to_job', 'job'))
    g.add_edges(job_to_machine_link_src, job_to_machine_link_dst, etype=('job', 'to_machine', 'machine'))

    return g


class heteroGNN(nn.Module):
    def __init__(self, Graph: torch.tensor, job_output_features_number: int, machine_output_features_number: int):
        super(heteroGNN, self).__init__()
        self.Graph = Graph
        numberOfJobs, numberOfMachines = Graph.shape[0], Graph.shape[1] - Graph.shape[0]
        self.job_output_features_number, self.machine_output_features_number = job_output_features_number, machine_output_features_number

        self.conv = dglnn.HeteroGraphConv({
                    'to_job': dglnn.GraphConv(5, job_output_features_number),
                    'to_machine': dglnn.GraphConv(5, machine_output_features_number)},
                    aggregate='mean')

        """
        Linear code here
        """
        self.linear_for_job = nn.Linear(job_output_features_number, Graph.shape[1])
        self.linear_for_machine = nn.Linear(numberOfMachines * machine_output_features_number,
                                            numberOfJobs * Graph.shape[1])


    def forward(self, h: list, L: list, W, P, N):
        numberOfJobs, numberOfMachines = len(h), self.Graph.shape[1] - len(h)
        jobs = list(zip(h, L))
        jobList = torch.tensor(sorted(jobs, key=lambda x: x[0], reverse=True))
        otherFeatures = torch.tensor([W, P, N])
        jobFeatures = torch.cat((jobList, otherFeatures.unsqueeze(1).t().expand(jobList.size(0), -1)), dim=1)  # n * 5

        machineFeatures = self.Graph[:, numberOfJobs:]  # n * m

        heteroGraph = tensor_to_dgl(self.Graph)

        features = {'job': jobFeatures,
                    'machine': machineFeatures}

        Graph_conv_re = self.conv(heteroGraph, features)
        job_conv, machine_conv = Graph_conv_re['job'], Graph_conv_re['machine'].flatten()

        Value = self.linear_for_job(job_conv) + self.linear_for_machine(machine_conv).view(numberOfJobs, -1)
        Possibility_tmp = Value - (1 - self.mask(self.Graph)) * 1e5
        Possibility = F.softmax(Possibility_tmp.view(-1), dim=0)
        Possibility = Possibility.view(Possibility_tmp.size())

        return Value, Possibility


    def mask(self, Graph: torch.tensor):
        left = torch.ones((Graph.size()[0], Graph.size()[0]))
        right = torch.ones((Graph.size()[0], Graph.size()[1]-Graph.size()[0]))
        row = torch.sum(Graph, dim=1, keepdim=True)
        col = torch.sum(Graph, dim=0, keepdim=True)

        left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
        left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
        left = left.triu(diagonal=1)
        right -= row

        return torch.cat((left, right), dim=1)


dd = heteroGNN(torch.tensor([[0,1,0,0,0,0],
                             [0,0,0,0,1,0],
                             [0,0,0,0,0,1],
                             [0,0,0,0,0,1]]), 3, 3)

print(dd.mask(dd.Graph))
print(dd.forward([1,2,3,4],[5,4,3,2],10,15,20))
