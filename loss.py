from utils import edge_index_to_adjacency_matrix
import torch
def caculate_loss(output_X, edge_index,device):
    num_nodes=len(output_X)
    true_adj=edge_index_to_adjacency_matrix(edge_index=edge_index, num_nodes=num_nodes).to(device)
    pred_adj=torch.mm(output_X, torch.transpose(output_X,0,1), out=None)

    loss=torch.mean(torch.pow(true_adj-pred_adj, 2))
    return loss, true_adj, pred_adj

def caculate_loss_p(x, output_X, edge_index, device, alpha):
    num_nodes=len(output_X)
    true_adj=edge_index_to_adjacency_matrix(edge_index=edge_index, num_nodes=num_nodes).to(device)
    pred_adj=torch.mm(output_X, torch.transpose(output_X,0,1), out=None)

    diff_attribute = torch.pow(x - output_X, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)
    
    loss= alpha * torch.mean(torch.pow(true_adj-pred_adj, 2)) + (1-alpha) * attribute_cost
    return loss, true_adj, pred_adj

