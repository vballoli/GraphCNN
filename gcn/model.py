import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Based on https://arxiv.org/abs/1609.02907

    Args:
        in_dims (int): Input feature dimensions.
        out_dims (int): Output feature dimensions.
        bias (bool): Include bias parameter in the forward method.
    """

    def __init__(self, in_dims, out_dims, bias=True):
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.w = nn.Parameter(torch.FloatTensor(in_dims, out_dims))
        sd = 1.0 / torch.sqrt(out_dims)
        self.w.data.uniform_(-sd, sd)
        if bias:
            self.b = nn.Parameter(torch.FloatTensor(out_dims))
            self.b.data.uniform_(-sd, sd)

    def forward(self, input_tensor, adjacent):
        """
        f(H, W) = sigmoid
        """
        output = torch.matmul(input_tensor, self.w)
        output = torch.matmul(adjacent, output)
        if self.b:
            output += b
        return output


class GCN(nn.Module):
    """
    Implementing an N layer Graph Convolution Network. (N excluding Input and Output layers). 
    All layers are followed by ReLU activation except for the last layer(since classification, log_softmax is used)

    Args:
        input_feature_dims (int):
        output_feature_dims (list(int)):
        hidden_dims (int): 
    """

    def __init__(self, input_feature_dims, output_feature_dims, hidden_dims):
        self.modules = nn.ModuleList()
        self.modules.append(GCNLayer(input_feature_dims, hidden_dims[0]))
        for i in range(1, len(n_layers-1)):
            self.modules.append(GCNLayer(hidden_dims[i], hidden_dims[i+1]))

    def forward(self, x, adj):
        for module in self.modules[:-1]:
            x = F.relu(module(x, adj))
        return F.log_softmax(self.modules[-1](x))
