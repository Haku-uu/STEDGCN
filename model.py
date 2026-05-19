from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
try:
    import pywt
except ImportError:
    pywt = None
from torch.nn import ModuleList
from stedgcn_attention import TemporalMultiHeadAttention

class DiffusiveGraphSignalPropagation(nn.Module):
    def __init__(self, channels=128, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.diffusion_step = diffusion_step
        self.conv = nn.Conv2d(diffusion_step * channels, channels, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        out = []
        for i in range(0, self.diffusion_step):
            if adj.dim() == 3:
                x = torch.einsum("bcnt,bnm->bcmt", x, adj).contiguous()
                out.append(x)
            elif adj.dim() == 2:
                x = torch.einsum("bcnt,nm->bcmt", x, adj).contiguous()
                out.append(x)
        x = torch.cat(out, dim=1)
        x = self.conv(x)
        output = self.dropout(x)
        return output

class SpatialAttentionPropagationLayer(nn.Module):

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpatialAttentionPropagationLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.matmul(h, self.W)  
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.transpose(2, 3)
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class SpatialAttentionPropagation(nn.Module):
    def __init__(self, n_in, n_out, dropout, alpha, nheads, order=1):
        super(SpatialAttentionPropagation, self).__init__()
        self.dropout = dropout
        self.nheads = nheads
        self.order = order

        self.attentions = [SpatialAttentionPropagationLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        for k in range(2, self.order + 1):
            self.attentions_2 = ModuleList(
                [SpatialAttentionPropagationLayer(n_in, n_out, dropout=dropout, alpha=alpha, concat=True) for _ in
                 range(nheads)])

        self.out_att = SpatialAttentionPropagationLayer(n_out * nheads * order, n_out, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)
        x = F.dropout(x, self.dropout, training=self.training)
        for k in range(2, self.order + 1):
            x2 = torch.cat([att(x, adj) for att in self.attentions_2], dim=-1)
            x = torch.cat([x, x2], dim=-1)
        x = F.elu(self.out_att(x, adj))
        return x    
    
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GatedFeatureSelector(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GatedFeatureSelector, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class PointwiseFeatureProjector(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(PointwiseFeatureProjector, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TemporalContextEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalContextEmbedding, self).__init__()

        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)

        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):

        day_emb = x[..., 1]  
        time_day = self.time_day[
            (day_emb[:, -1, :] * self.time).type(torch.LongTensor)
        ]  
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        week_emb = x[..., 2]  
        time_week = self.time_week[
            (week_emb[:, -1, :]).type(torch.LongTensor)
        ]  
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        return time_day, time_week


class PreNormalization(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):

        return self.fn(self.norm(x), **kwargs)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)



class TrendTemporalAttentionUnit(nn.Module):
    def __init__(self, c_in, num_nodes, tem_size):
        super(TrendTemporalAttentionUnit, self).__init__()


    def forward(self, seq):

        return x_1


class FrequencySpecificTemporalDynamicsExtractor(nn.Module):
    def __init__(self, features=128, layers=4, length=12, num_nodes=170, dropout=0.1):
        super(FrequencySpecificTemporalDynamicsExtractor, self).__init__()


        self.low_freq_layers = nn.ModuleList([
            TrendTemporalAttentionUnit(features, num_nodes, length) for _ in range(layers)
        ])

        kernel_size = int(length / layers + 1)
        self.high_freq_layers = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(features, features, (1, kernel_size)),
            nn.ReLU(),
            nn.Dropout(dropout)) for _ in range(layers)
        ])

        self.alpha = nn.Parameter(torch.tensor(-5.0))

    def forward(self, XL, XH):
        res_xl = XL
        res_xh = XH

        for layer in self.low_freq_layers:
            XL = layer(XL)
        
        XL = (res_xl[..., -1] + XL[..., -1]).unsqueeze(-1)

        XH = nn.functional.pad(XH, (1, 0, 0, 0))  
        

        for layer in self.high_freq_layers:
            XH = layer(XH)  

        XH = (res_xh[..., -1] + XH[..., -1]).unsqueeze(-1)

        alpha_sigmoid = torch.sigmoid(self.alpha)
        output = alpha_sigmoid * XL + (1 - alpha_sigmoid) * XH

        return output 
 
class DualPathGraphDependencyLayer(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(DualPathGraphDependencyLayer, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  # We assume d_v always equals d_k
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model

        
        self.gcn = DiffusiveGraphSignalPropagation(channels=256, diffusion_step=1, dropout=dropout)
        self.gat = SpatialAttentionPropagation(256, 256, dropout, alpha=0.2, nheads=1)
        
        self.LayerNorm = LayerNorm(
            [d_model, num_nodes, seq_length], elementwise_affine=False
        )
        self.dropout1 = nn.Dropout(p=dropout)
        self.gated_feature_selector = GatedFeatureSelector(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.alpha = nn.Parameter(torch.tensor(-5.0)) 
        self.weight = nn.Parameter(torch.ones(256, self.num_nodes, 1))   
        self.bias = nn.Parameter(torch.zeros(256, self.num_nodes, 1))
        
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 6).to(device), requires_grad=True).to(device)
        self.nodevec2 = nn.Parameter(torch.randn(6, num_nodes).to(device), requires_grad=True).to(device)
        

    def forward(self, input, D_Graph):
        #print('input', input.shape)        #input torch.Size([64, 256, 170, 1])        

        A_graph = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1).unsqueeze(0)
        x_gcn = self.gcn(input, A_graph)

        x_gat = self.gat(input.transpose(1,3), D_Graph).transpose(1,3)


        alpha_sigmoid = torch.sigmoid(self.alpha)  
        x =  alpha_sigmoid* x_gat +  (1 - alpha_sigmoid) * x_gcn
        
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.gated_feature_selector(x) + x
        x = x * self.weight + self.bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        

        return x
    
    
class DualPathGraphDependencyConstructor(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length, dropout, num_layers):
        super(DualPathGraphDependencyConstructor, self).__init__()
        
        
        self.layers = nn.ModuleList([
            DualPathGraphDependencyLayer(device, 
                    d_model=d_model, 
                    head=head, 
                    num_nodes=num_nodes, 
                    seq_length=seq_length, 
                    dropout=dropout)
            for _ in range(num_layers)  
        ])

    def forward(self, x, D_Graph):
        #print(x.shape)
        for layer in self.layers:
            x = layer(x, D_Graph)
        return x


class EventAwareGraphConstructor(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        adj_dyn_1 = torch.softmax(
            F.relu(
                torch.einsum("bcnt, cm->bnm", x, self.memory).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_dyn_2 = torch.softmax(
            F.relu(
                torch.einsum("bcn, bcm->bnm", x.sum(-1), x.sum(-1)).contiguous()
                / math.sqrt(x.shape[1])
            ),
            -1,
        )
        adj_f = torch.cat([(adj_dyn_1).unsqueeze(-1)] + [(adj_dyn_2).unsqueeze(-1)], dim=-1)

        adj_f = torch.softmax(self.fc(adj_f).squeeze(), -1)

        topk_values, topk_indices = torch.topk(adj_f, k=int(adj_f.shape[1] * 0.8), dim=-1)

        mask = torch.zeros_like(adj_f)

        mask.scatter_(-1, topk_indices, 1)

        adj_f = adj_f * mask

        return adj_f


class EventAwareGraphConvolutionNetwork(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = EventAwareGraphConstructor(channels, num_nodes, diffusion_step, dropout)
        self.gcn = DiffusiveGraphSignalPropagation(channels, diffusion_step, dropout)
        self.emb = emb

    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x

class TrendAdaptiveGraphConvolutionNetwork(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1, emb=None):
        super().__init__()

        self.conv = nn.Conv2d(channels,channels,(1,1))
        self.generator = TrendAdaptiveGraphConstructor(channels, num_nodes, diffusion_step, dropout)
        self.gcn = DiffusiveGraphSignalPropagation(channels, diffusion_step, dropout)
        self.emb = emb


    def forward(self, x):

        skip = x
        x = self.conv(x)
        adj_dyn = self.generator(x)
        x = self.gcn(x, adj_dyn)
        x = x*self.emb + skip

        return x

class TrendAdaptiveGraphConstructor(nn.Module):
    def __init__(self, channels=128, num_nodes=170, diffusion_step=1, dropout=0.1):
        super().__init__()
        self.memory = nn.Parameter(
            torch.randn(channels, num_nodes))
        nn.init.xavier_uniform_(self.memory)
        self.fc = nn.Linear(2, 1)
        self.E_adaptive = nn.Parameter(torch.randn(num_nodes, 10))

    def forward(self, x):

        return adj_f
class GatedFeatureSelector(nn.Module):
    def __init__(self, features, dropout=0.1):#PEMS08: 192
        super(GatedFeatureSelector, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):


        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out#[64,192,170,12]

class TemporalCausalConvolution(nn.Module):
    def __init__(self, features, kernel_size=2, dropout=0.2, levels=1):
        super(TemporalCausalConvolution, self).__init__()

        layers = []
        for i in range(levels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(features, features, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.chomp = CausalChomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, xh):
        xh = self.tcn(xh)
        return xh
    pass


class CausalChomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(CausalChomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class STEDGCN(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=883,
        input_len=12,
        output_len=12,
        dropout=0.1,
    ):
        super().__init__()

        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        self.layers = 2
        self.dims = 6
        if num_nodes == 883:
            self.input_dim = 3

        if num_nodes == 883 or num_nodes == 170 or num_nodes == 304 or num_nodes == 307 or num_nodes == 325 or num_nodes == 207:
            time = 288
            self.layers = 2
        elif num_nodes == 358 :
            time = 288
            self.layers = 1

        self.temporal_context_embedding = TemporalContextEmbedding(time, channels)
        self.start_conv_res = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_1 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.start_conv_2 = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.network_channel = channels * 2
        
        self.frequency_specific_temporal_dynamics_extractor = FrequencySpecificTemporalDynamicsExtractor(
            features = 128, 
            layers = self.layers, 
            length = self.input_len, 
            num_nodes = self.num_nodes, 
            dropout=0.1
        )
        
        
        self.dual_path_graph_dependency_constructor = DualPathGraphDependencyConstructor(
            device,
            d_model = self.network_channel,
            head = self.head,
            num_nodes = num_nodes,
            seq_length = 1,
            dropout = dropout,
            num_layers = self.layers
        )
        
        
        self.graph_context_projector = nn.Conv2d(
            in_channels=3,
            out_channels=self.dims,
            kernel_size=(1, 1)
        )

        self.spatial_event_prior = nn.Parameter(torch.randn(64, self.dims, num_nodes, 1).to(device), requires_grad=True).to(device)
        
        self.day_projection = nn.Conv2d(channels, self.dims, kernel_size=(1, 1))
        self.week_projection = nn.Conv2d(channels, self.dims, kernel_size=(1, 1))
        
        self.spatiotemporal_projection = nn.Conv2d(
            self.network_channel, self.network_channel, kernel_size=(1, 1)
        )

        self.trajectory_forecast_head = nn.Conv2d(
            channels, self.output_len, kernel_size=(1, self.output_len)
        )


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def _temporal_signal_separator(self, input_data):
       
        if pywt is not None:
            residual_numpy = input_data.detach().cpu().numpy()
            coef = pywt.wavedec(residual_numpy, 'db1', level=2)
            coefl = [coef[0]] + [None] * (len(coef) - 1)
            coefh = [None] + coef[1:]
            xl = pywt.waverec(coefl, 'db1')
            xh = pywt.waverec(coefh, 'db1')
            trend = torch.from_numpy(xl).to(self.device).type_as(input_data)
            event = torch.from_numpy(xh).to(self.device).type_as(input_data)
            return trend[..., :input_data.shape[-1]], event[..., :input_data.shape[-1]]

        batch_size, channels, num_nodes, seq_len = input_data.shape
        flattened = input_data.reshape(batch_size * channels * num_nodes, 1, seq_len)
        low_frequency = flattened
        for _ in range(2):
            low_frequency = F.avg_pool1d(low_frequency, kernel_size=2, stride=2, ceil_mode=True)
        trend = F.interpolate(low_frequency, size=seq_len, mode='linear', align_corners=False)
        trend = trend.reshape(batch_size, channels, num_nodes, seq_len)
        event = input_data - trend
        return trend, event

    def forward(self, history_data):
        

        return prediction
