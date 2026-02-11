import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

import numpy as np
import torchvision.transforms.functional
import torch
import torch.nn as nn
from models import Dit1D

from models.wav2vec2 import Wav2Vec2Model
from models.base_model import BaseModel
# from models.hubert import HubertModel
from models.wavlm import WavLMModel
from models.TCN import TCN
import torch.nn.functional as F

from einops import rearrange
from diffusers.models.attention import BasicTransformerBlock
from models.Dit1D import TimestepEmbedder



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
    """
    MyModel:
        * 156 dim -> (100 + 50 + 6) 
        * audio -> (32,512)
        * dit input: (22+32=54, 512)
        * output: (b,70)

    Args:
        nn (_type_): _description_
    """
    def __init__(self, cfg):
        super(MyModel, self).__init__()
        self.cfg = cfg
        self.interpolate_pos = cfg['interpolate_pos']

        if 'wav2vec2' in cfg['audio_encoder_repo']:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(cfg['audio_encoder_repo'])
            self.audio_encoder_dim = 1024
        elif 'wavlm' in cfg['audio_encoder_repo']:
            self.audio_encoder = WavLMModel.from_pretrained(cfg['audio_encoder_repo'])
            self.audio_encoder_dim = 768
        else:
            raise ValueError("wrong audio_encoder_repo")
        
        self.audio_encoder.feature_extractor._freeze_parameters()

        if cfg['freeze_wav2vec']:
            self.audio_encoder._freeze_wav2vec2_parameters()


        self.n_layers = cfg['n_layers']
        self.max_time_steps = cfg['max_time_steps']
        self.num_frames = cfg['num_frames']
        self.n_heads = cfg['n_heads']
        self.dropout = cfg['dropout']
        self.dit_dm = cfg['dit_dm']
        self.tcn = TCN(cfg['flame_dim'], self.audio_encoder_dim) 

        self.t_embedder = TimestepEmbedder(6*self.audio_encoder_dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        
        # self.latent_in_proj = nn.Linear(156, self.audio_encoder_dim)

        # self.shape_in_proj = nn.Linear(100, self.audio_encoder_dim)
        # self.exp_in_proj = nn.Linear(50, self.audio_encoder_dim)
        # self.pose_in_proj = nn.Linear(6, self.audio_encoder_dim)

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

        self.final_proj = OutHead(in_dim=self.audio_encoder_dim, d_ff=2*self.dit_dm, out_dim=156, head_dropout=self.dropout)

        # self.shape_out_proj = OutHead(in_dim=self.dit_dm, d_ff=2*self.dit_dm, out_dim=100, head_dropout=self.dropout)
        # self.exp_out_proj = OutHead(in_dim=self.dit_dm, d_ff=2*self.dit_dm, out_dim=50,  head_dropout=self.dropout)
        # self.pose_out_proj = OutHead(in_dim=self.dit_dm, d_ff=2*self.dit_dm, out_dim=6,  head_dropout=self.dropout)
        
    
    def forward(self, latent, audio, timestep, fps=None):
        """_summary_

        Args:
            latent (_type_) and source_motion_emb: (b, f,latent_dim)    # (b, 500, 156)
            audio (_type_): (b,f,audio_len)                             # (320000, )  
            optional : first_frame                                      # (b, 1, 156)
            timestep (_type_): (b,), torch.long                         # (b, )
        Returns:
            _type_: (b, output_dim)
        """
        if fps is None:
            frame_num = latent.shape[1]  # b, f, d
        else:
            frame_num = round(audio.shape[-1]/16000 * fps)
        
        audio_hidden_states = self.audio_encoder(audio, frame_num=frame_num, interpolate_pos = self.interpolate_pos) 

        audio_enc_hidden_states = audio_hidden_states.last_hidden_state   # ([b, f, 768]) 

        t = self.t_embedder(timestep)    # [4, 4608]

        x_latent = self.tcn(latent)          # ([b, f, 768]) 

        for block in self.blocks:
            x_latent = block(hidden_states=x_latent, encoder_hidden_states=audio_enc_hidden_states, timestep=t)
    

        # [latent_shape, latent_exp, latent_pose,] -> [b, f, 3* d]
        # audio  [b, 2*f, 3d]
        # latent_inputs = torch.concat([latent_shape, latent_exp, latent_pose, audio_enc], dim=-1) # [b, f, 4*d]
        # hidden_states = self.one_net(latent, t=timestep, condition=audio_enc)


        # latent_shape_output = self.shape_out_proj(hidden_states) 
        # latent_exp_output = self.exp_out_proj(hidden_states)
        # latent_pose_output = self.pose_out_proj(hidden_states)
        # latent_output =  torch.concat([latent_shape_output, latent_exp_output, latent_pose_output],  dim=-1) #[b, f, 156]
        
        latent_output = self.final_proj(x_latent)

        return latent_output
    

if __name__ == "__main__":
    import argparse, yaml
    
    def update_cfg_from_argv(cfg, argv):
        # assuming cfg is your dict object and argv = sys.argv
        for i, arg in enumerate(argv[1:]):
            # Assuming key and value are separated by '='        
            key, value = arg.split('=')
            assert key.startswith("--")
            key = key[len("--"):]
            try:
                cfg[key] = eval(value)
            except:
                cfg[key] = str(value)
            
        return cfg

    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--config",
        type=str,
        default="/scratch3/wan451/3DTalk/Talking3DHead/config/default_config.yaml",
    )
    
    args, _ = parser.parse_known_args()
    
    # load cfg and update it w.r.t argv
    with open(args.config) as fid:
        cfg = yaml.load(fid, Loader=yaml.Loader)
    cfg = update_cfg_from_argv(cfg=cfg, argv=sys.argv)

    x_latent = torch.rand((4, 500, 156))
    x_audio = torch.rand((4, 320000))
    x_first_frame = x_latent[:, 0, :]  # (1,1,156)
    x_timestep =  torch.rand((4,))    #(b, )

    model = MyModel(cfg=cfg)
    y = model(latent=x_latent, audio=x_audio, timestep=x_timestep, fps=25)
    print(y.shape)