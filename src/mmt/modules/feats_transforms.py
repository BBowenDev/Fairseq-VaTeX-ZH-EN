import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding
)
from fairseq.modules.quant_noise import quant_noise

class FeatsTransformFlatten(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before        

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[0]
        self.feats_dim = len(self.feats_shape)
        
        if self.feats_dim == 1:
            self.linear = nn.Linear(self.feats_channels, self.embed_dim)
            self.avgpool = None
        elif self.feats_dim == 3:
            self.linear = nn.Conv2d(self.feats_channels, self.embed_dim, kernel_size=3)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        else:
            raise(Exception("feats_dim"))

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, x):
        x = self.dropout_module(x)
        x = self.activation_fn(self.linear(x))
        x = self.activation_dropout_module(x)
        if self.avgpool is not None:
            x = self.avgpool(x)
        x = x.reshape((-1,1,self.embed_dim))
        x = self.fc(x)
        x = self.dropout_module(x)
        return x


class FeatsTransform(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before        

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[0]
        self.feats_dim = len(self.feats_shape)
        
        if self.feats_dim == 1:
            self.linear = nn.Linear(self.feats_channels, self.embed_dim)
        elif self.feats_dim == 3:
            self.linear = nn.Conv2d(self.feats_channels, self.embed_dim, kernel_size=3)
        else:
            raise(Exception("feats_dim"))

        self.fc = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, x):
        x = self.dropout_module(x)
        x = self.activation_fn(self.linear(x))
        x = self.activation_dropout_module(x)
        if self.feats_dim == 3:
            x = x.reshape((x.shape[0],self.embed_dim,-1)).permute((0,2,1))
        else:
            x = x.reshape((x.shape[0],1,self.embed_dim))
        x = self.fc(x)
        x = self.dropout_module(x)
        return x

    
class FeatsTransformLinear(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before        

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[0]
        self.feats_dim = len(self.feats_shape)
        
        self.fc = nn.Linear(self.feats_channels, self.embed_dim)

    def forward(self, x):
        x = self.dropout_module(x)
        if self.feats_dim == 3:
            x = x.reshape((x.shape[0],self.feats_channels,-1)).permute((0,2,1))
        else:
            x = x.reshape((x.shape[0],1,self.feats_channels))
        x = self.fc(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        return x


class FeatsTransformPosEmb(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[0]
        self.feats_dim = len(self.feats_shape)
        
        self.feats_max_regions = 256
        self.embed_positions = PositionalEmbedding(
            self.feats_max_regions,
            self.embed_dim,
            padding_idx=0)

        self.fc1 = nn.Linear(self.feats_channels, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, x):
        x = self.dropout_module(x)
        if self.feats_dim == 3:
            x = x.reshape((x.shape[0],self.feats_channels,-1)).permute((0,2,1))
        else:
            x = x.reshape((x.shape[0],1,self.feats_channels))
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        e = self.embed_positions(torch.ones(x.shape[0], x.shape[1], device=x.device))
        x = x + e
        x = self.fc2(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        return x


class FeatsTransformPosEmbLinear(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[0]
        self.feats_dim = len(self.feats_shape)

        if self.feats_dim == 1:
            self.feats_max_regions = 1
            self.embed_positions = None
        elif self.feats_dim == 3:
            self.feats_max_regions = self.feats_shape[1] * self.feats_shape[2]
            self.embed_positions = PositionalEmbedding(
                self.feats_max_regions,
                self.feats_channels,
                padding_idx=0)

        self.fc = nn.Linear(self.feats_channels, self.embed_dim)

    def forward(self, x):
        x = self.dropout_module(x)
        if self.feats_dim == 3:
            x = x.reshape((x.shape[0],self.feats_channels,-1)).permute((0,2,1))
            e = self.embed_positions(torch.ones(x.shape[0], x.shape[1], device=x.device))
            x = x + e
        else:
            x = x.reshape((x.shape[0],1,self.feats_channels))
        
        x = self.fc(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        return x
    

    

class FeatsTransformSimpleLinear(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, 'activation_fn', 'relu') or "relu"
        )
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        self.feats_shape = eval(args.feats_shape)        
        self.feats_channels = self.feats_shape[1]
        self.feats_dim = len(self.feats_shape)

        self.fc = nn.Linear(self.feats_channels, self.embed_dim)

    def forward(self, x):
        x = self.dropout_module(x)
        x = self.fc(x)
        x = self.activation_fn(x)
        x = self.activation_dropout_module(x)
        return x
    

    
