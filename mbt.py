import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import MyMMT_utils as utils
from self_attention_pooling import attention_pooling
import os
from ir import IR50
from mobilefacenet import MobileFaceNet
from TIMNET import TIMNET
from poster import PosterV2

os.environ['TORCH_USE_CUDA_DSA'] = '1'

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoding[:, :x.size(1), :]
        e = e.to(device)
        x = x.to(device)
        x = x + e
        return x


class MLPBlock(nn.Module):
    def __init__(self, mlp_dim, dropout_rate=0.1, out_dim=None, use_bias=True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.dense1 = nn.Linear(mlp_dim, mlp_dim*4, bias=use_bias)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(mlp_dim*4, out_dim if out_dim is not None else mlp_dim, bias=use_bias)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        out_dim = self.out_dim if self.out_dim is not None else x.shape[-1]
        self.dense2.out_features = out_dim
        x = F.gelu(self.dense1(x))
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, mlp_dim, num_heads, dropout_rate=0.5, attention_dropout_rate=0.5,is_cuda = True):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate

        self.norm1 = nn.LayerNorm(normalized_shape=mlp_dim)
        self.attn = nn.MultiheadAttention(embed_dim=mlp_dim, num_heads=num_heads, dropout=attention_dropout_rate)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.norm2 = nn.LayerNorm(normalized_shape=mlp_dim)
        self.fc1 = nn.Linear(mlp_dim, mlp_dim * 4)
        self.act = nn.GELU()
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(mlp_dim * 4, mlp_dim)
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.is_cuda = is_cuda

        if self.is_cuda:
            self.cuda()


    def forward(self, inputs):

        # Attention block.
        x = self.norm1(inputs)
        x = x.permute(1, 0, 2)  # Convert to (seq_len, batch_size, embed_dim)

        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, embed_dim)
        x = self.dropout1(x)

        x = x + inputs

        # MLP block.
        y = self.norm2(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.dropout2(y)
        y = self.fc2(y)
        y = self.dropout3(y)

        y = y + x

        return y


class Encoder(torch.nn.Module):
    def __init__(self, mlp_dim: int, num_layers: int, num_heads: int,
                 dropout_rate: float = 0.5, attention_dropout_rate: float = 0.5,
                 stochastic_droplayer_rate: float = 0.0, modality_fusion: Tuple[str] = ('audio', 'text'),
                 fusion_layer: int = 0, use_bottleneck: bool = True,
                 test_with_bottlenecks: bool = True, share_encoder: bool = False,is_cuda = True):
        super().__init__()

        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.stochastic_droplayer_rate = stochastic_droplayer_rate
        self.modality_fusion = modality_fusion
        self.fusion_layer = fusion_layer
        self.use_bottleneck = use_bottleneck
        self.test_with_bottlenecks = test_with_bottlenecks
        self.share_encoder = share_encoder

        # Define the layers
        self.encoder_blocks = {}
        self.encoder_norm = nn.LayerNorm(normalized_shape=mlp_dim*3)

        for lyr in range(self.num_layers):
            droplayer_p = (lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
            self.encoder_blocks[lyr] = {}
            for modality in self.modality_fusion:
                if self.share_encoder and modality != self.modality_fusion[0]:
                    self.encoder_blocks[lyr][modality] = self.encoder_blocks[lyr][self.modality_fusion[0]]
                else:
                    self.encoder_blocks[lyr][modality] = EncoderBlock(mlp_dim=self.mlp_dim,
                                                                      num_heads=self.num_heads,
                                                                      dropout_rate=self.dropout_rate,
                                                                      attention_dropout_rate=self.attention_dropout_rate)

        self.is_cuda = is_cuda

        if self.is_cuda:
            self.cuda()

    def forward(self, x: Dict[str, Any], bottleneck: torch.Tensor, train: bool):
        def get_context(target_modality, modality_fusion, x):
            context = []
            for modality in modality_fusion:
                if modality != target_modality and modality in modality_fusion:
                    context.append(x[modality])
            return context

        def combine_context(x, other_modalities):
            if len(x.shape) == 4:
                num_tokens = x.shape[2]
            elif len(x.shape) == 2:
                num_tokens = x.shape[0]
            else:
                num_tokens = x.shape[1]
            x = torch.reshape(x, [1, 1, num_tokens, self.mlp_dim])  # 转换为 [1, 1, n, 768] 形状
            x = x[0]
            if len(other_modalities[0].shape) == 4:
                other_modalities[0] = other_modalities[0][0]
            elif len(other_modalities[0].shape) == 2:
                other_modalities[0] = torch.reshape(other_modalities[0],(1,other_modalities[0].shape[0],768))

            x = x.to(device)
            other_modalities.append(x)
            other_modalities[0] = other_modalities[0].to(device)
            x_combined = torch.cat(other_modalities, dim=1)
            return x_combined, num_tokens

        use_bottlenecks = train or self.test_with_bottlenecks
        x_combined = None

        for lyr in range(self.num_layers):
            droplayer_p = (lyr / max(self.num_layers - 1, 1)) * self.stochastic_droplayer_rate
            dropout_mask = (torch.rand(x[next(iter(self.modality_fusion))].size(0),
                                       device=x[next(iter(self.modality_fusion))].device) > droplayer_p).float()

            if (lyr < self.fusion_layer or len(self.modality_fusion) == 1 or
                    (self.use_bottleneck and not use_bottlenecks)):
                for modality in self.modality_fusion:
                    x[modality] = self.encoder_blocks[lyr][modality](x[modality])* dropout_mask.unsqueeze(-1)
            else:
                if self.use_bottleneck:

                    # 将outputf与原有的bottleneck相加
                    bottleneck = bottleneck
                    bottle = []
                    for modality in self.modality_fusion:
                        t_mod = x[modality].shape[1]
                        in_mod = torch.cat([x[modality], bottleneck], dim=1)
                        out_mod = self.encoder_blocks[lyr][modality](in_mod)
                        x[modality] = out_mod[:, :t_mod]
                        bottle.append(out_mod[:, t_mod:])
                    bottleneck = torch.mean(torch.stack(bottle, dim=-1), dim=-1)
                else:
                    if not self.share_encoder and len(self.modality_fusion) > 1:
                        x_new = {}
                        for modality in self.modality_fusion:
                            other_modalities = get_context(modality, self.modality_fusion, x)
                            combined_mods, t = combine_context(x[modality], other_modalities)
                            combined_mods = self.encoder_blocks[lyr][modality](combined_mods)
                            x_new[modality] = combined_mods[:, -t:]
                        x = x_new

                    elif self.share_encoder and len(self.modality_fusion) > 1:
                        if x_combined is None:
                            x_combined = []
                            for modality in self.modality_fusion:
                                x_combined.append(x[modality])
                            x_combined = torch.cat(x_combined, dim=1)
                        x_combined = self.encoder_blocks[lyr][self.modality_fusion[0]](x_combined)

        if x_combined is not None:
            x_out = x_combined
        else:
            x_out = []
            for modality in self.modality_fusion:
                x_out.append(x[modality][:, 0, :])
            x_out = torch.cat(x_out, dim=1)
        encoded = self.encoder_norm(x_out)
        return encoded


class MBT(nn.Module):
    def __init__(self, mlp_dim: int = 768, num_layers: int = 4, num_heads: int = 8, num_classes: int = 4, patches: Dict[str, Any] = {},
                 hidden_size: int = 768, representation_size: Optional[int] = None, dropout_rate: float = 0.5, attention_dropout_rate: float = 0.5,
                 stochastic_droplayer_rate: float = 0., classifier: str = 'token', modality_fusion: Tuple[str] = ('audio','video','text'),
                 fusion_layer: int = 2, return_prelogits: bool = False, return_preclassifier: bool = False,
                 use_bottleneck: bool = True, n_bottlenecks: int = 3, test_with_bottlenecks: bool = True,
                 share_encoder: bool = False, dtype: Any = torch.float32,is_cuda = True):
        super().__init__()

        # 定义子模块和参数
        # self.temporal_encoding_config = temporal_encoding_config
        self.patches = patches
        self.hidden_size = hidden_size
        self.encoder = Encoder(mlp_dim=mlp_dim, num_layers=num_layers, num_heads=num_heads,
                               dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate,
                               stochastic_droplayer_rate=stochastic_droplayer_rate, modality_fusion=modality_fusion,
                               fusion_layer=fusion_layer, use_bottleneck=use_bottleneck,
                               test_with_bottlenecks=test_with_bottlenecks, share_encoder=share_encoder)
        self.classifier_type = classifier
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.return_prelogits = return_prelogits
        self.return_preclassifier = return_preclassifier
        self.use_bottleneck = use_bottleneck
        self.n_bottlenecks = n_bottlenecks
        self.test_with_bottlenecks = test_with_bottlenecks
        self.modality_fusion = modality_fusion

        if representation_size is not None:
            self.pre_logits = nn.Linear(in_features=mlp_dim, out_features=representation_size)

        self.output_projection = nn.Linear(in_features=representation_size if representation_size else mlp_dim,
                                           out_features=num_classes, bias=False)
        self.linear = LinearLayer(self.hidden_size*3,self.hidden_size)
        self.classifier = LinearLayer(self.hidden_size, 4)
        self.softmax = nn.Softmax(dim=1)
        self.is_cuda = is_cuda

        if self.is_cuda:
            self.cuda()

    def forward(self, audio,video,text,  train: bool,debug: bool = False):

        x = {"audio": audio, "video":video, "text": text}
        class_token = torch.zeros(1, 1, self.hidden_size, device=x[self.modality_fusion[0]].device)
        for modality in self.modality_fusion:
            a = class_token.repeat(x[modality].size(0), 1, 1).to(device)
            b = x[modality]
            b = torch.reshape(b, (1, b.shape[1], self.hidden_size)).to(device)
            x[modality] = torch.cat((a, b), dim=1)
        # 使用给定的 Encoder 进行编码
        if self.use_bottleneck:
            bottleneck = torch.zeros(x[self.modality_fusion[0]].size(0), self.n_bottlenecks, self.hidden_size,
                                     device=x[self.modality_fusion[0]].device)
        else:
            bottleneck = None

        encoded = self.encoder(x, bottleneck,train)

        # encoded = encoded[:,0,:]

        # 计算注意力权重

        logits = self.linear(encoded)
        logits = F.gelu(logits)
        logits = self.classifier(logits)
        logits = torch.reshape(logits,(1,4))
        logits = F.softmax(logits,dim=1)

        return logits

class MBTClassificationModel(nn.Module):
    def __init__(self,is_cuda = True):
        super().__init__()
        self.model = MBT()
        self.num_classes = 4
        self.is_cuda = is_cuda

    def forward(self, audio: torch.Tensor,video:torch.Tensor, text: torch.Tensor, label:torch.Tensor,train: bool = True):

        logits = self.model(audio,video,text,train)

        return logits



def attention_pooling(x):
    """
    Attention Pooling for a tensor x of shape [batch_size, 1, seq_length, hidden_size]
    """
    # Flatten the input tensor
    x_flat = x.view(x.size(0), -1, x.size(-1))  # [batch_size, seq_length, hidden_size]

    # Compute attention weights for each time step
    attn_weights = torch.softmax(torch.matmul(x_flat, x_flat.transpose(1, 2)), dim=-1)  # [batch_size, seq_length, seq_length]

    # Apply attention weights to the flattened tensor
    x_attended = torch.matmul(attn_weights, x_flat)  # [batch_size, seq_length, hidden_size]

    # Compute the attention pooled output
    output = torch.mean(x_attended, dim=1)  # [batch_size, hidden_size]

    return output


class DynamicMBT(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout,modality_fusion: Tuple[str] = ('audio','video','text'),):
        super().__init__()
        self.views = 2
        self.in_dim = in_dim
        self.num_classes = num_class
        self.dropout = dropout
        self.modality_fusion = modality_fusion
        self.l2_decay_factor = 0.0025
        # self.pos_enc = PositionalEncoding(d_model=768, max_len=1000)  # Add positional encoding
        self.mbt = MBTClassificationModel()
        self.low_rank = LinearLayer(64,16)
        self.mobilefacenet = MobileFaceNet()
        self.ir = IR50()
        self.TimNet = TIMNET(nb_filters=64,dilations=12)
        self.PostV2 = PosterV2(self.mobilefacenet,self.ir)
        self.is_cuda = True

        if self.is_cuda:
            self.cuda()
            self.TimNet.cuda()
            self.PostV2.cuda()

    def forward(self, audio,video, text, label, train=True):
        # audio = self.pos_enc(audio)
        # text = self.pos_enc(text)
        audio = self.TimNet(audio)
        audio = torch.reshape(audio,[1,64,768])
        audio = audio.permute(0,2,1)
        audio = self.low_rank(audio)
        audio = audio.permute(0,2,1)
        video = self.PostV2(video)
        video = torch.reshape(video,[1,video.shape[0],768])
        # 使用MBT分类器
        mbt_logits = self.mbt(audio,video,text, label,train)
        mbt_loss = self.loss_function(mbt_logits,label,self.parameters())

        return mbt_logits , mbt_loss


    def loss_function(self, logits: torch.Tensor, label: torch.Tensor, model_params):
        weights = None
        sof_ce_loss = utils.weighted_softmax_cross_entropy(logits, label, weights,
                                                           label_smoothing=None,
                                                           )
        l2_loss = torch.sum(torch.square(torch.cat([x.flatten() for x in model_params])))
        total_loss = sof_ce_loss + 0.5 * self.l2_decay_factor * l2_loss

        return total_loss

def wsce(logits: torch.Tensor, targets: torch.Tensor,
                                    weights: Optional[torch.Tensor] = None,
                                    label_smoothing: Optional[float] = None) -> torch.Tensor:

    num_classes = logits.size(-1)
    one_hot_targets = targets

    # Optionally apply label smoothing.
    if label_smoothing is not None:
        on_value = 1.0 - label_smoothing
        off_value = label_smoothing / num_classes
        one_hot_targets = one_hot_targets * on_value + off_value

    log_probs = F.log_softmax(logits, dim=-1)
    loss = -torch.sum(one_hot_targets * log_probs, dim=-1)

    if weights is not None:
        loss *= weights

    return torch.mean(loss)

class add_positional_embed(nn.Module):

    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoding[:, :x.size(1), :]
        e = e.to(device)
        x = x.to(device)
        x = x + e
        return x


def pad_or_truncate(tensor, target_len):
    """
    Pad or truncate the input tensor to a target length.

    :param tensor: The tensor to be modified, expected shape is [1, seq_len, 768].
    :param target_len: The target length for the sequence dimension.
    :return: The modified tensor with shape [1, target_len, 768].
    """
    seq_len = tensor.shape[1]

    if seq_len == target_len:
        return tensor
    elif seq_len < target_len:
        pad_size = target_len - seq_len
        pad_tensor = torch.zeros((1, pad_size, 768), dtype=tensor.dtype).to(tensor.device)
        return torch.cat((tensor, pad_tensor), dim=1)
    else:  # seq_len > target_len
        return tensor[:, :target_len, :]