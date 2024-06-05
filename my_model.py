import torch
import torch.nn as nn
import torchvision as ts
import torch_geometric.nn as tg
import pyro

def make_mlp_model(latent_dimension,
                   input_dimension,
                   num_layers):
    layers = [latent_dimension] * (num_layers - 1)
    layers.append(input_dimension)
    mlp=nn.Sequential(ts.ops.MLP(in_channels=1, hidden_channels=layers))
    return mlp

def make_gnn_model(args,hidden_dim,out_dim):
    GNN_MAP={
    'GAT':
    tg.models.GAT(in_channels=-1,
                  hidden_channels=hidden_dim,
                  num_layers=args.num_layer,
                  out_channels=out_dim),
    'GCN':
    tg.models.GCN(in_channels=-1,
                  hidden_channels=hidden_dim,
                  num_layers=args.num_layer,
                  out_channels=out_dim),
    'GIN':
    tg.models.GIN(in_channels=-1,
                  hidden_channels=hidden_dim,
                  num_layers=args.num_layer,
                  out_channels=out_dim),
    'GraphUNet':
    tg.models.GraphUNet(in_channels=-1,
                      hidden_channels=hidden_dim,
                      depth=args.num_layer,
                      out_channels=out_dim),
    'GraphSAGE':
    tg.models.GraphSAGE(in_channels=-1,
                      hidden_channels=hidden_dim,
                      num_layers=args.num_layer,
                      out_channels=out_dim)
    }
    return GNN_MAP[args.gnn_name]

class NF(nn.Module):
    def __init__(self,
                 args,
                 use_batch_norm=False,
                 weight_sharing=False,
                 name="GRevNet"):
        super().__init__()
        self.name=name
        self.num_timesteps = args.num_timesteps
        self.weight_sharing = weight_sharing
        self.gnn_counter = 1
        self.bn_counter = 1
        if weight_sharing:
            self.s = []
            self.t = []
            for _ in range(2):
                self.s.append(getattr(self, self.create_gnn(args)))
                self.t.append(getattr(self, self.create_gnn(args)))
        else:
            self.s = []
            self.t = []
            for _ in range(2):
                tempL1=[]
                tempL2=[]
                for _ in range(self.num_timesteps):
                    tempL1.append(getattr(self, self.create_gnn(args)))
                    tempL2.append(getattr(self, self.create_gnn(args)))
                self.s.append(tempL1)
                self.t.append(tempL2)
        self.use_batch_norm = use_batch_norm
        self.bns = []
        for _ in range(2):
            tempL1=[]
            for _ in range(self.num_timesteps):
                    tempL1.append(getattr(self, self.create_bn(args)))
            self.bns.append(tempL1)
    
    def create_gnn(self,args):
        var_name=f"gnn{self.gnn_counter}"
        setattr(self, var_name, make_gnn_model(args, hidden_dim=args.hidden_dim1,out_dim=int(args.output_dim/2)))
        self.gnn_counter += 1
        return var_name

    def create_bn(self,args):
        var_name=f"bn{self.gnn_counter}"
        setattr(self,
                var_name,
                pyro.distributions.transforms.BatchNorm(int(args.hidden_dim/2)))
        self.bn_counter += 1
        return var_name

    
    def F(self, data, edge_index):
        log_det_jacobian = 0
        xx=torch.chunk(data, chunks=2, dim = 1)
        x0, x1 = xx[0], xx[1]
        for i in range(self.num_timesteps):
            if self.use_batch_norm:
                bn = self.bns[0][i]
                log_det_jacobian += bn.log_abs_det_jacobian(x=x0, y=bn(x0))
                x0 = bn._inverse(x1)
            if self.weight_sharing:
                s = self.s[0](x0, edge_index)
                t = self.t[0](x0, edge_index)
            else:
                s = self.s[0][i](x0, edge_index)
                t = self.t[0][i](x0, edge_index)
            s=torch.sigmoid(s)
            t=torch.sigmoid(t)
            log_det_jacobian += torch.sum(s)
            x1 = x1* torch.exp(s) + t

            if self.use_batch_norm:
                bn = self.bns[1][i]
                log_det_jacobian += bn.log_abs_det_jacobian(x=x1, y=bn(x1))
                x1 =bn._inverse(x1)
            if self.weight_sharing:
                s = self.s[1](x1, edge_index)
                t = self.t[1](x1, edge_index)
            else:
                s = self.s[1][i](x1, edge_index)
                t = self.t[1][i](x1, edge_index)
            s=torch.sigmoid(s)
            t=torch.sigmoid(t)
            log_det_jacobian += torch.sum(s)
            x0 = x0 * torch.exp(s) + t
        x =torch.cat((x0, x1), dim=1)
        return x, log_det_jacobian
    
    def G(self, dataz, edge_indexz):
        zz=torch.chunk(dataz, chunks=2, dim = 1)
        z0, z1 = zz[0], zz[1]
        for i in reversed(range(self.num_timesteps)):
            if self.weight_sharing:
                s = self.s[1](z1, edge_indexz)
                t = self.t[1](z1, edge_indexz)
            else:
                s = self.s[1][i](z1, edge_indexz)
                t = self.t[1][i](z1, edge_indexz)
            if self.use_batch_norm:
                bn = self.bns[1][i]
                z1 = bn(z1)
            s=torch.sigmoid(s)
            t=torch.sigmoid(t)
            z0 = (z0 - t) * torch.exp(-s)

            if self.weight_sharing:
                s = self.s[0](z0, edge_indexz)
                t = self.t[0](z0, edge_indexz)
            else:
                s = self.s[0][i](z0, edge_indexz)
                t = self.t[0][i](z0, edge_indexz)
            if self.use_batch_norm:
                bn = self.bns[0][i]
                z0 = bn(z0)
            s=torch.sigmoid(s)
            t=torch.sigmoid(t)
            z1 = (z1 - t) * torch.exp(-s)
        return torch.cat((z0, z1), dim=1)

    def log_prob(self, x):
        z, log_det_jacobian = self.F(x)
        return torch.sum(self.prior.log_prob(z)) + log_det_jacobian

    def forward(self, x, edge_index, inverse=True):
        func = self.F if inverse else self.G
        return func(x,edge_index)



