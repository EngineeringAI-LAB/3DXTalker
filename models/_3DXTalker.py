import math
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import numpy as np
import torchvision.transforms.functional
import torch
import torch.nn as nn

from models.wav2vec2 import Wav2Vec2Model
from models.base_model import BaseModel
# from models.hubert import HubertModel
from models.wavlm import WavLMModel
from models.TCN import TCN
import torch.nn.functional as F
import math
from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock
from funasr import AutoModel


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_channels, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_channels, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

# linear interpolation layer
def linear_interpolation(features, output_len: int):
    features = features.transpose(1, 2)
    output_features = F.interpolate(
        features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class OutHead(nn.Module):
    def __init__(self, in_dim: int, d_ff: int, out_dim: int, head_dropout: float):
        super().__init__()
      
        self.head_layer = nn.Sequential(
            nn.Linear(in_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, out_dim)
        )
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        return self.dropout(self.head_layer(x))
    

class MyModel(nn.Module):
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.cfg = cfg
        self.interpolate_pos = cfg['interpolate_pos']

        if 'wav2vec2' in cfg['audio_encoder_repo']:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg['audio_encoder_repo'], use_safetensors=True)
            self.audio_encoder_dim = 1024
        elif 'wavlm' in cfg['audio_encoder_repo']:
            self.audio_encoder = WavLMModel.from_pretrained(cfg['audio_encoder_repo'], use_safetensors=True)
            self.audio_encoder_dim = 768
        else:
            raise ValueError("wrong audio_encoder_repo")
        
        self.audio_encoder.feature_extractor._freeze_parameters()

        if cfg['freeze_wav2vec']:
            self.audio_encoder._freeze_wav2vec2_parameters()

        self.emo2vec = AutoModel(model="./pretrained_models/emotion2vec_plus_base",
                                disable_update=True,
                                device=str(cfg['device']),
                                )
        
        for param in self.emo2vec.model.parameters():
            param.requires_grad = False

        
        self.n_layers = cfg['n_layers']

        self.n_heads = cfg['n_heads']
        self.dropout = cfg['dropout']

        self.t_embedder = TimestepEmbedder(6*self.audio_encoder_dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        self.latent_in_proj = nn.Linear(cfg['flame_dim'], self.audio_encoder_dim)
        self.refer_in_proj = nn.Linear(cfg['flame_dim'], self.audio_encoder_dim)

        self.latent_layerNorm = nn.LayerNorm(self.audio_encoder_dim)
        self.envelope_layerNorm = nn.LayerNorm(self.audio_encoder_dim)
        self.emofea_layerNorm = nn.LayerNorm(self.audio_encoder_dim)


        self.envelop_encoder= nn.Sequential(
                                    nn.Conv1d(640, 512, kernel_size=3, dilation=1, padding=1),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Conv1d(512, self.audio_encoder_dim, kernel_size=3, dilation=2, padding=2),
                                    nn.BatchNorm1d(self.audio_encoder_dim),
                                    nn.ReLU(),
                                    nn.Conv1d(self.audio_encoder_dim, self.audio_encoder_dim, kernel_size=3, dilation=4, padding=4),
                                    nn.BatchNorm1d(self.audio_encoder_dim),
                                    nn.ReLU(),
                                    nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
                                )

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(dim=self.audio_encoder_dim,
                                  num_attention_heads=self.n_heads,
                                  attention_head_dim = self.audio_encoder_dim//self.n_heads,
                                  dropout = self.dropout,
                                  cross_attention_dim = self.audio_encoder_dim,
                                  positional_embeddings = "sinusoidal",
                                  norm_type="ada_norm_single",
                                  num_positional_embeddings=1000,
                                  ) for _ in range(self.n_layers)
                                ])

        self.identity_head_layers = nn.ModuleList([
            BasicTransformerBlock(dim=self.audio_encoder_dim,
                                  num_attention_heads=self.n_heads,
                                  attention_head_dim = self.audio_encoder_dim//self.n_heads,
                                  dropout = self.dropout,
                                  cross_attention_dim = None,
                                  double_self_attention=True,
                                  positional_embeddings = "sinusoidal",
                                  norm_type="ada_norm_single",
                                  num_positional_embeddings=1000,
                                  ) for _ in range(1)
                                ])
        
        self.pose_head_layers = nn.ModuleList([
            BasicTransformerBlock(dim=self.audio_encoder_dim,
                                  num_attention_heads=self.n_heads,
                                  attention_head_dim = self.audio_encoder_dim//self.n_heads,
                                  dropout = self.dropout,
                                  cross_attention_dim = self.audio_encoder_dim,
                                  positional_embeddings = "sinusoidal",
                                  norm_type="ada_norm_single",
                                  num_positional_embeddings=1000,
                                  ) for _ in range(2)
                                ])
        
        self.exp_head_layers = nn.ModuleList([
            BasicTransformerBlock(dim=self.audio_encoder_dim,
                                  num_attention_heads=self.n_heads,
                                  attention_head_dim = self.audio_encoder_dim//self.n_heads,
                                  dropout = self.dropout,
                                  cross_attention_dim = self.audio_encoder_dim,
                                  positional_embeddings = "sinusoidal",
                                  norm_type="ada_norm_single",
                                  num_positional_embeddings=1000,
                                  ) for _ in range(2)
                                ])

        self.exp_proj = OutHead(in_dim=self.audio_encoder_dim, d_ff=2*self.audio_encoder_dim, out_dim=50, head_dropout=self.dropout)
        self.pose_proj = OutHead(in_dim=self.audio_encoder_dim, d_ff=2*self.audio_encoder_dim, out_dim=6, head_dropout=self.dropout)
        self.identity_proj = OutHead(in_dim=self.audio_encoder_dim, d_ff=2*self.audio_encoder_dim, out_dim=228, head_dropout=self.dropout)

    
    def forward(self, latent, x0, audio, envelope, timestep, fps=None):
        """_summary_

        Args:
            latent (_type_) and source_motion_emb: (b, t, d)    # (b, t, d)
            audio (_type_): (b, audio_len)                             # (b, n)  
            refer_frame :  (b, 1, d)                                    # (b, 1, d)
            envelope : (b, audio_len)                                  # (b, n) 
            timestep (_type_): (b,), torch.long                         # (b, )
        Returns:
            _type_: (b, t, d)
        """

        b, f, d = latent.shape
        if fps is None:
            frame_num = latent.shape[1]  # b, f, d
        else:
            frame_num = round(audio.shape[-1]/16000 * fps)
        
        audio_hidden_states = self.audio_encoder(audio, frame_num=frame_num, interpolate_pos = self.interpolate_pos) 
        audio_enc_hidden_states = audio_hidden_states.last_hidden_state   # ([b, f, 768]) 

        envelope = envelope.reshape((b, frame_num, -1))    # (b, n, 640) 
        envelope = envelope.permute(0, 2, 1)            # (b, c, t)
        envelope_latent = self.envelop_encoder(envelope)
        envelope_latent = envelope_latent.permute(0, 2, 1)

        envelope_latent = self.envelope_layerNorm(envelope_latent)
        
        emotion_fea_from_audio = self.emo2vec.model.extract_features(audio)['x']
        emotion_fea_from_audio = F.interpolate(emotion_fea_from_audio.transpose(1, 2),
                                            size=frame_num,
                                            mode="linear",
                                            align_corners=True
                                        ).transpose(1, 2)  # [B, t, 768]
        emotion_fea_from_audio = self.emofea_layerNorm(emotion_fea_from_audio)

        t = self.t_embedder(timestep)   

        delta_latent = self.latent_in_proj(latent)         # ([b, t, 768]) 
        x0_latent = self.refer_in_proj(x0)                # ([b, 1, 768])
        
        x_latent = x0_latent + delta_latent
        x_latent = self.latent_layerNorm(x_latent)

        for block in self.blocks:
            x_latent = block(hidden_states=x_latent, encoder_hidden_states=audio_enc_hidden_states, timestep=t)
    
        pose_latent = x_latent
        for layer in self.pose_head_layers:
            pose_latent = layer(hidden_states=pose_latent, encoder_hidden_states=envelope_latent, timestep=t)

        exp_latent = x_latent
        for layer in self.exp_head_layers:
            exp_latent = layer(hidden_states=exp_latent, encoder_hidden_states=emotion_fea_from_audio, timestep=t)

        identity_latent = x_latent
        for layer in self.identity_head_layers:
            identity_latent = layer(hidden_states=identity_latent, encoder_hidden_states=None, timestep=t)
        
        x_exp = self.exp_proj(exp_latent)
        x_pose = self.pose_proj(pose_latent)
        x_identity = self.identity_proj(identity_latent)
        x_shape = x_identity[:, :, :100]  
        x_detail = x_identity[:, :, 100:] 
    
        latent_output = torch.cat([x_shape, x_exp, x_pose, x_detail], dim=-1)

        return latent_output
    

if __name__ == "__main__":
    import argparse, yaml

    def update_cfg_from_args(cfg, args):
        args_dict = vars(args)
        for k, v in args_dict.items():
            if k in cfg and v is not None:
                cfg[k] = v
            if k not in cfg:
                cfg[k] = v
        return cfg
    
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="/scratch3/wan451/3DTalk/The-Sound-of-Motion/config/default_config.yaml",
    )
    
    args, _ = parser.parse_known_args()
    
    # load cfg and update it w.r.t argv
    with open(args.config) as fid:
        cfg = yaml.load(fid, Loader=yaml.Loader)

    cfg = update_cfg_from_args(cfg, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg['device'] = device


    x_latent = torch.rand((4, 500, 284)).to(device)
    x_audio = torch.rand((4, 320000)).to(device)
    x_0 = x_latent[:, 0:1, :]  # (1,1,156)
    x_diff = x_latent - x_0
    x_timestep =  torch.rand((4,)).to(device)    #(b, )
    x_envelope = torch.rand((4, 320000)).to(device)

    model = MyModel(cfg=cfg).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    y = model(latent=x_diff, x0=x_0, audio=x_audio, envelope= x_envelope, timestep=x_timestep, fps=25)
    print(y.shape)

    