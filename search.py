def calculate_time_nodummy(Graph):
    G_tmp = Graph[:, :, 0:self.numberOfJobs]
    row = torch.transpose(torch.sum(G_tmp, dim=2, keepdim=True), 1, 2)
    col = torch.sum(G_tmp, dim=1, keepdim=True)

    totalTime_OMA_fake = (1 - row - col) * self.T_list
    totalTime_NOMA_fake = torch.sum(G_tmp * self.T, dim=1, keepdim=True)
    totalTime_fake = totalTime_OMA_fake + totalTime_NOMA_fake

    Time_b_1_n = torch.bmm(totalTime_fake, (Graph[:, :, self.numberOfJobs: Graph.size()[2]]))
    max_values, _ = torch.max(Time_b_1_n, dim=2)

    return torch.squeeze(max_values, dim=1)


def mask(Graph: torch.tensor):
    left = torch.ones((Graph.size()[0], Graph.size()[0]))
    right = torch.ones((Graph.size()[0], Graph.size()[1] - Graph.size()[0]))
    row = torch.sum(Graph, dim=1, keepdim=True)
    col = torch.sum(Graph, dim=0, keepdim=True)

    left = left - row - torch.t(row) - col[:, 0:Graph.size()[0]] - torch.t(col[:, 0:Graph.size()[0]])
    left = torch.where(left == 1, torch.tensor(1).float(), torch.tensor(0).float())
    left = left.triu(diagonal=1)
    right -= row

    return torch.cat((left, right), dim=1)

def optimal_solution_and_value(Graph)
    mask = mask(Graph)
    if torch.sum(mask) == 0:
        return calculate_time_nodummy(Graph), Graph

    for mask in all_mask:
        newGraph  = Graph + mask
        optimal_time, optimal_Graph = optimal_solution_and_value(newGraph)
