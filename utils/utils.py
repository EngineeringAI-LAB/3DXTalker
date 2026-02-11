import argparse
import logging
import numpy as np
import os
import random
import torch
from copy import deepcopy
import json
from scipy import signal

def get_audio_encoder_dim(audio_encoder: str):
    if audio_encoder == 'microsoft/wavlm-base-plus':
        return 768
    elif audio_encoder == 'facebook/wav2vec2-large-xlsr-53':
        return  1024
    else:
        raise ValueError("wrong audio_encoder")
    
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def get_logger(log_file: str):
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode='w')
    fmt = '[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]=>%(message)s'  # noqa: E501
    handler.setFormatter(logging.Formatter(fmt))
    for hdl in logger.handlers:
        logger.removeHandler(hdl)
    logger.addHandler(handler)
    return logger


def log_datasetloss(
    logger,
    epoch: int,
    phase: str,
    loss_dict: dict
):
    base_str = f'{phase.upper():<5} Epoch: {epoch:03d} '

    for loss_type, avg_meter in loss_dict.items():
        base_str += f'{loss_type}_:{avg_meter.avg: .8f}, '
    logger.info(base_str)
    return base_str


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self, ):
        return self.avg


def get_average_meter_dict(phase):
    if phase == 'train':
        metric_name_list = ['velocity_loss']
    elif phase == 'val' or phase == 'test':
        metric_name_list = ['rec_L2_loss', 'LVE', 'LVD']
    else:
        raise ValueError('wrong phase for average meter')
    average_meter_dict_template = {}
    for k in metric_name_list:
        average_meter_dict_template[k] = AverageMeter()
    # mixed_dataset_name = 'mixed_dataset'
    # dataset_name_list = args.dataset + [mixed_dataset_name]
    # average_meter_dict = {}
    # for dataset_name in dataset_name_list:
    #     average_meter_dict[dataset_name] = deepcopy(
    #         average_meter_dict_template)
    return average_meter_dict_template, metric_name_list

def write_to_tensorboard(writer,
                         epoch: int,
                         phase: str,
                         loss_dict: dict,
                        ):

    for loss_type, avg_meter in loss_dict.items():
        key = f'{phase}/{loss_type}'
        writer.add_scalar(key, avg_meter.avg, epoch)
    return 0

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



def dump_jsonl(output_fn, list_obj):
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    with open(output_fn, "a") as fid:
        for x in list_obj:
            a_str = json.dumps(x)
            fid.write(a_str + "\n")


def batch_orth_proj(X, camera):
    ''' orthgraphic projection
        X:  3d vertices, [bz, n_point, 3]
        camera: scale and translation, [bz, 3], [scale, tx, ty]
    '''
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:,:,2:]], 2)
    shape = X_trans.shape
    Xn = (camera[:, :, 0:1] * X_trans)
    return Xn


def wav_interpolate(audio_data, sr=16000, fps=25):
    if audio_data.shape[-1]%sr != 0:
        expected_size = int(round(audio_data.shape[-1]/sr) * fps * (sr/fps))
        audio_data = signal.resample(audio_data, expected_size)
    return audio_data

    
def wrap_postprocess(preds, audio_data, sr=16000, fps=25):
    audio_data = audio_data.squeeze()
    audio_data = wav_interpolate(audio_data, sr=16000, fps=25)
    audio_algin_frame = audio_data.reshape((round(audio_data.shape[-1]/sr * fps), -1)) # [500, 640]
    audio_algin_frame = audio_algin_frame.abs().mean(dim=-1)  # (f, )
    ad_min, ad_max = audio_algin_frame.min(), audio_algin_frame.max()
    normed_audio = (audio_algin_frame-ad_min) / (ad_max-ad_min)
    lamcof = 4
    beta_e = lamcof*normed_audio.pow(1)
    delta = 0.2
    coef = torch.abs(preds)/delta  
    out = delta * (1 - torch.exp(- beta_e * coef))
    return out


def hilbert_envelope_torch(signal):
    # signal: [B, T]
    orig_dtype = signal.dtype
    signal = signal.to(torch.float32)
    fft = torch.fft.fft(signal, dim=-1)
    N = signal.shape[-1]

    h = torch.zeros(N, device=signal.device)
    h[0] = 1
    if N % 2 == 0:
        h[1:N//2] = 2
        h[N//2] = 1
    else:
        h[1:(N+1)//2] = 2

    H = fft * h
    analytic = torch.fft.ifft(H, dim=-1)
    envelope = torch.abs(analytic)
    return envelope.to(orig_dtype)


def init_seg_interp(batch_size=4, T=250, D=284, seg_len=25, first_frame_latent=None, device=None, dtype=None):

    S = (T + seg_len - 1) // seg_len

    keys = torch.randn(batch_size, S + 1, D, device=device, dtype=dtype)  # [B, S+1, D]
    if first_frame_latent is not None:
        # first_frame_latent
        keys[:, 0:1, :] = first_frame_latent

    parts = []
    for s in range(S):
        n = min(seg_len, T - s * seg_len)
        t = torch.linspace(0, 1, n, device=device, dtype=dtype).view(1, n, 1)  # [1, n, 1]
        t = t.repeat(batch_size, 1, 1) #[bs, n, 1]
        seg = (1 - t) * keys[:, s:s+1, :] + t * keys[:, s+1:s+2, :] # [B, n, D]
        parts.append(seg)

    latent = torch.cat(parts, dim=1)  # [B, T, D]
    return latent

def audio_align_frame(audio_data, sr=16000, fps=25):
    # audio_data = audio_data.squeeze()
    b, wav_len = audio_data.shape
    audio_data = wav_interpolate(audio_data, sr=16000, fps=25)
    audio_algin_frame = audio_data.reshape((b, int(wav_len/sr * fps), int(sr/fps))) # [f, 640]
    audio_algin_frame = audio_algin_frame.abs().mean(dim=-1)  # (f, )
    # smooth
    ad_min, ad_max = audio_algin_frame.min(), audio_algin_frame.max()
    normed_audio = (audio_algin_frame-ad_min) / (ad_max-ad_min)
    return normed_audio

def init_smoothed_noise(batch_size, n_motion_frames=250, latent_dim=284, device=None, dtype=None, envelope=None, lip_cof=1, first_frame_latent=None, noise_type="seg_interp_25", use_envelope=True):

    if noise_type=="random":
        noise_latents = torch.randn([batch_size, n_motion_frames, latent_dim], device=device, dtype=dtype)
    elif noise_type=="seg_interp_25":
        noise_latents = init_seg_interp(batch_size, n_motion_frames, latent_dim, 25, first_frame_latent, device=device, dtype=dtype)
    elif noise_type=="seg_interp_250":
        noise_latents = init_seg_interp(batch_size, n_motion_frames, latent_dim, 250, first_frame_latent, device=device, dtype=dtype)
    end_latent = noise_latents[:,-1:,:].clone()
    if use_envelope:
        normed_envelope = audio_align_frame(envelope)
        noise_latents[:, :, 153] = torch.abs(noise_latents[:, :, 153])*normed_envelope*lip_cof
    return noise_latents, end_latent