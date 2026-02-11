import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy import signal

def wav_interpolate(audio_data, sr=16000, fps=25):
    if audio_data.shape[-1]%sr != 0:
        expected_size = int(int(audio_data.shape[-1]/sr) * fps * (sr/fps))
        audio_data = signal.resample(audio_data, expected_size, axis=-1)
    return audio_data

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


def plot_audio_envlp(input_audio_data, envelope_input, smoothed_envlp, wav_f, split_idx):
    wav_name, wav_ext = os.path.splitext(wav_f)
    import matplotlib.pyplot as plt
    start_audio_idx = int(10*16000)
    end_audio_idx = int(11*16000)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(input_audio_data[0].cpu().numpy()[start_audio_idx:end_audio_idx], label='Audio', color="tab:red", linewidth=1.5)
    ax.plot(envelope_input[0].cpu().numpy()[start_audio_idx:end_audio_idx], label='Envelope', color="tab:blue", linewidth=1.5)
    ax.plot(smoothed_envlp[0].cpu().numpy()[start_audio_idx:end_audio_idx], label='SmoothedEnvelope', color="tab:orange", linewidth=1.5)
    ax.legend()
    ax.set_title(f"{wav_f}")
    fig_out_dir = f'/scratch3/wan451/3DTalk/The-Sound-of-Motion/render_outputs/evlp_plots'
    os.makedirs(fig_out_dir, exist_ok=True)
    out_fig_path = os.path.join(fig_out_dir, f'{wav_name}_{split_idx}.png')
    plt.savefig(out_fig_path)
    plt.close()

def plot_velocity(pred_velocity, noisy_latents, initial_noisy_latents, tau):
    plt.close()
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pred_velocity[0,:,153].detach().cpu().numpy(), label='pred_velocity', color="tab:red", linewidth=1.5)
    ax.plot(noisy_latents[0,:,153].detach().cpu().numpy(), label='noisy_latents', color="tab:blue", linewidth=1.5)
    ax.plot(initial_noisy_latents[0,:,153].detach().cpu().numpy(), label='initial_noisy_latents', color="tab:orange", linewidth=1.5)
    ax.set_ylim(-4,4)
    ax.legend(loc='upper left')
    fig_out_path = '/scratch3/wan451/3DTalk/The-Sound-of-Motion/render_outputs/noisy_latent_Vary'
    os.makedirs(fig_out_path, exist_ok=True)
    fig_name = os.path.join(fig_out_path, f'{tau:03d}.png')
    # plt.show()
    plt.savefig(fig_name)


def plot_pose3_diff(pose3_diff_sequences, envelope_input, pred_pose3_abs_before_post, pred_pose3_abs_after_post, wav_f, split_idx, idendity_name, cfg):
    normed_envelope = audio_align_frame(envelope_input)
    normed_envelope_smooth = savgol_filter(normed_envelope[0].cpu().numpy(), window_length=25, polyorder=2)
    wav_name, wav_ext = os.path.splitext(wav_f)
    gt_annotation_seq = np.load(os.path.join(cfg['data_root'], idendity_name, idendity_name+"-AllInOne.npy"))
    gt_x0 = gt_annotation_seq[0]
    gt_annot_delta = gt_annotation_seq - gt_x0
    gt_pose3_delta = gt_annot_delta[:, 153]
    gt_pose3_abs = gt_annotation_seq[:, 153]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(pose3_diff_sequences[0].cpu().numpy(), label='pred_pose3_offset', color="tab:gray", linewidth=1.5)
    # ax.plot(pred_pose3_abs_before_post.cpu().numpy(), label='pred_pose3_abs_before_post', color="tab:pink", linewidth=1.5)
    ax.plot(pred_pose3_abs_after_post.cpu().numpy(), label='pred_pose3_abs_after_post', color="tab:green", linewidth=1.5)
    # ax.plot(normed_envelope.cpu().numpy(), label='Envelope', color="tab:blue", linewidth=1.5)
    # ax.plot(normed_envelope_smooth, label='SmoothedEnvelope', color="tab:cyan", linewidth=1.5)
    ax.plot(gt_pose3_delta[:500], label='true_pose3_offset', color="tab:orange", linewidth=1.5)
    ax.plot(gt_pose3_abs[:500], label='true_pose3_abs', color="tab:purple", linewidth=1.5)
    ax.legend()
    ax.set_title(f"{wav_f}")
    fig_out_dir = f'/scratch3/wan451/3DTalk/The-Sound-of-Motion/render_outputs/FrameDelta_plots'
    os.makedirs(fig_out_dir, exist_ok=True)
    out_fig_path = os.path.join(fig_out_dir, f'{wav_name}_{split_idx}.png')
    plt.savefig(out_fig_path)
    plt.close()


def save_obj(refer_img_name, wav_name, detail_verts, frame_idx):
    obj_save_path = f'/scratch3/wan451/3DTalk/The-Sound-of-Motion/render_outputs/demo_preds-{refer_img_name}/objs'
    os.makedirs(obj_save_path, exist_ok=True)
    obj_save_file = os.path.join(obj_save_path, f'DetailVerts_RefImg-{refer_img_name}_Audio-{wav_name}_Frame-{frame_idx:04d}.obj')
    from decalib.utils import util
    dense_faces = np.load('/scratch3/wan451/3DTalk/DECA2/data/my_dense_faces_6.npy') 
    if dense_faces.min() == 1:
        dense_faces = dense_faces-1
    util.write_obj(obj_save_file, 
        detail_verts[frame_idx], 
        dense_faces,
        # colors = dense_colors,
        inverse_face_order=True)