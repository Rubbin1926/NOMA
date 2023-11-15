import dgl
import torch
from torch import nn
from torch.nn import functional as F
from dgl.nn import TAGConv


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

    g = dgl.heterograph({('job', 'to', 'job'): ([], []),
                         ('job', 'to', 'machine'): ([], [])})

    g.add_edges(job_to_job_link_src, job_to_job_link_dst, etype=('job', 'to', 'job'))
    g.add_edges(job_to_machine_link_src, job_to_machine_link_dst, etype=('job', 'to', 'machine'))

    return g


re = tensor_to_dgl(torch.tensor([[0,1,0,0,0,0],
                                 [0,0,0,0,1,0],
                                 [0,0,0,0,0,1],
                                 [0,0,0,0,0,1]]))
print(re)

