import numpy as np
import librosa
from numpy.typing import NDArray


def generate_flame_global_pose_from_audio(
    audio_data: np.ndarray,
    fps: int = 25,
    sr: int = 16000,
    style: str = "presentation",
    # 角度上限（度）
    yaw_max_deg: float = 16.0,
    pitch_max_deg: float = 10.0,
    roll_max_deg: float = 6.0,
    # 平滑/节奏参数
    smoothing_sec: float = 0.35,   # 低通平滑
    pause_fade_sec: float = 0.25,  # 静音衰减到中性位的时间常数
    # 随机微动
    jitter_deg: float = 0.5,
    seed: int | None = 0,
    euler_order: str = "zyx",
    return_features: bool = False,
) -> tuple[NDArray[np.float32], dict] | NDArray[np.float32]:
    """
    根据演讲音频，生成 FLAME global rotation 的轴角轨迹 rotvec，形状 [T, 3]（弧度）。
    - yaw: 左右（默认绕 z）
    - pitch: 点头（默认绕 y）
    - roll: 歪头（默认绕 x）
    欧拉顺序默认 'zyx'（roll->pitch->yaw 复合）。

    Parameters
    ----------
    audio_path : str
        音频文件路径（任意 librosa 可读格式）。
    fps : int
        目标逐帧采样率（与视频/网格渲染一致）。
    sr : int
        加载音频重采样采样率。
    style : str
        'presentation' 目前仅此风格，决定幅度与频率范围。
    yaw_max_deg, pitch_max_deg, roll_max_deg : float
        三轴角度上限（度）。
    smoothing_sec : float
        角度信号最终平滑时间窗（秒）。
    pause_fade_sec : float
        静音/停顿时回到中性位的指数衰减时间常数（秒）。
    jitter_deg : float
        叠加的随机微动强度（度，已会被再次平滑）。
    seed : int | None
        随机种子（None 则不设种子）。
    euler_order : str
        欧拉角顺序，默认 'zyx'（与 yaw/pitch/roll 对应 z/y/x）。
    return_features : bool
        若 True，同时返回中间特征（能量、F0、onset 等，按 fps 对齐）。

    Returns
    -------
    rotvec : np.ndarray [T,3], float32
        每帧轴角向量（Rodrigues），单位弧度。
    features (optional) : dict
        中间特征字典（按 fps 对齐），仅当 return_features=True。
    """
    # if seed is not None:
    #     np.random.seed(seed)
    y = audio_data.astype(float)
    # ========= 读取音频 & 基本特征 =========
    # y, sr = librosa.load(audio_path, sr=sr, mono=True)
    duration = len(audio_data) / sr
    T = int(np.ceil(duration * fps))
    t_fps = np.arange(T) / fps

    hop = int(sr / 100)  # 10ms 帧


    # 能量包络（RMS）
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop).flatten()
    rms = rms / (np.percentile(rms, 95) + 1e-8)
    rms = np.clip(rms, 0, 1.5)

    # F0（pyin），无声处为 NaN
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=sr, frame_length=2048, hop_length=hop
        )
    except Exception:
        # 备选：YIN（返回不为 NaN，但无声区也有数值）
        f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"),
                         fmax=librosa.note_to_hz("C7"), sr=sr,
                         frame_length=2048, hop_length=hop)
        # 近似无声掩码：RMS 过低
        f0[np.interp(np.arange(len(f0)), np.arange(len(rms)), rms) < 0.02] = np.nan

    # onset 强度与节拍
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    try:
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_env, sr=sr, hop_length=hop, tightness=100
        )
    except Exception:
        tempo, beat_frames = 120.0, np.array([], dtype=int)

    # ========= 对齐到 fps 时间轴 =========
    def interp_feat(x, x_times):
        x = np.asarray(x, dtype=float)
        x_times = np.asarray(x_times, float)
        assert len(x) == len(x_times), f"length mismatch: x={len(x)}, x_times={len(x_times)}"
        # NaN 前后向填充
        if np.isnan(x).any():
            x = forward_backward_fill_nan(x)
        # 时间轴边界保护
        x_times = np.asarray(x_times)
        if x_times[0] > 0:
            x_times = np.concatenate([[0.0], x_times])
            x = np.concatenate([[x[0]], x])
        if x_times[-1] < duration:
            x_times = np.concatenate([x_times, [duration]])
            x = np.concatenate([x, [x[-1]]])
        return np.interp(t_fps, x_times, x)

        # === 各自的时间轴 ===
    times_rms   = librosa.frames_to_time(np.arange(len(rms)),       sr=sr, hop_length=hop)
    times_onset = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)
    times_f0    = librosa.frames_to_time(np.arange(len(f0)),        sr=sr, hop_length=hop)

    # === 插值到 t_fps ===
    rms_i   = interp_feat(rms, times_rms)
    onset_i = interp_feat(onset_env / (np.percentile(onset_env, 95) + 1e-8), times_onset)

    f0_log  = np.log(np.clip(f0, 1e-6, None))
    f0_i    = interp_feat(f0_log, times_f0)
    voiced_i = interp_feat((~np.isnan(f0)).astype(float), times_f0)

    # 归一化/标准化的辅助量
    z_rms = zscore_safe(rms_i)
    z_onset = zscore_safe(onset_i)
    # F0 的微分反映语调起伏
    df0 = deriv_1d(f0_i)
    z_df0 = zscore_safe(df0)

    # ========= 将韵律映射为三轴角度（度） =========
    # 1) pitch：以能量与语调变化为主，演讲式点头
    pitch_raw = 0.65 * z_rms + 0.35 * z_df0
    pitch_raw = smooth_1d(pitch_raw, win=int(0.20 * fps) | 1)
    pitch_deg = soft_clip(pitch_raw, lim=2.5) * pitch_max_deg

    # 2) yaw：缓慢左右摆头 + 节拍/语速调制（随 onsets/tempo）
    base_sway_hz = 0.18 + 0.002 * np.clip(tempo, 60, 180)  # ~0.3-0.54Hz
    # 振幅因子：说话更强（onset / rms 高）时更大
    amp_mod = 0.55 + 0.25 * sigmoid(0.8 * z_onset) + 0.20 * sigmoid(0.6 * z_rms)
    # 相位随 onsets 微扰
    phase = 2 * np.pi * base_sway_hz * t_fps + 0.20 * np.cumsum(deriv_1d(z_onset))
    yaw_wave = np.sin(phase)
    yaw_deg = yaw_wave * yaw_max_deg * amp_mod
    # 轻微偏置（说多了会有慢速移位），但受限
    drift = cumsum_lowpass(0.003 * zscore_safe(np.random.randn(T)), alpha=0.995)
    yaw_deg += np.clip(drift * yaw_max_deg * 0.25, -yaw_max_deg * 0.25, yaw_max_deg * 0.25)

    # 3) roll：很小，一部分跟随 pitch（自然耦合），再加少量噪声
    roll_deg = 0.28 * pitch_deg + 0.10 * z_rms * roll_max_deg

    # ========= 停顿/静音处理：回中性位 & 抑制抖动 =========
    # 声音越弱/voiced 越低，衰减越强
    damping = 0.25 + 0.75 * np.clip(voiced_i, 0, 1)  # 0.25~1
    pitch_deg *= damping
    yaw_deg *= damping
    roll_deg *= (0.6 + 0.4 * damping)

    # 指数回中性（对静音尤为明显）
    lam = np.exp(-1.0 / max(1, int(pause_fade_sec * fps)))
    pitch_deg = exp_fade_to_zero(pitch_deg, lam)
    yaw_deg = exp_fade_to_zero(yaw_deg, lam)
    roll_deg = exp_fade_to_zero(roll_deg, lam)

    # ========= 加入小幅随机微动（再平滑） =========
    if jitter_deg > 0:
        jx = smooth_1d(np.random.randn(T), win=int(0.15 * fps) | 1) * jitter_deg
        jy = smooth_1d(np.random.randn(T), win=int(0.20 * fps) | 1) * jitter_deg
        jz = smooth_1d(np.random.randn(T), win=int(0.25 * fps) | 1) * jitter_deg
        yaw_deg += jy * 0.6
        pitch_deg += jx * 0.5
        roll_deg += jz * 0.4

    # ========= 终端平滑 & 限幅 =========
    win_final = max(3, int(smoothing_sec * fps) | 1)
    yaw_deg = smooth_1d(yaw_deg, win=win_final)
    pitch_deg = smooth_1d(pitch_deg, win=win_final)
    roll_deg = smooth_1d(roll_deg, win=win_final)

    yaw_deg = np.clip(yaw_deg, -yaw_max_deg, yaw_max_deg)
    pitch_deg = np.clip(pitch_deg, -pitch_max_deg, pitch_max_deg)
    roll_deg = np.clip(roll_deg, -roll_max_deg, roll_max_deg)

    # ========= 欧拉角(度)->轴角(弧度) =========
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)
    rotvec = euler_to_rotvec_batch(roll, pitch, yaw, order=euler_order).astype(np.float32)

    if return_features:
        feats = dict(
            t=t_fps, rms=rms_i, onset=onset_i, voiced=voiced_i,
            yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
            tempo=tempo, fps=fps
        )
        return rotvec, feats
    return rotvec


# ----------------- 工具函数 -----------------

def forward_backward_fill_nan(x: NDArray[np.floating]) -> NDArray[np.floating]:
    x = x.copy()
    # forward
    n = len(x)
    idx = np.where(~np.isnan(x))[0]
    if len(idx) == 0:
        return np.zeros_like(x)
    # 前向
    last = idx[0]
    for i in range(0, last):
        x[i] = x[last]
    for i in range(last + 1, n):
        if np.isnan(x[i]):
            x[i] = x[i - 1]
    # 反向再补
    nextv = x[np.where(~np.isnan(x))[0][-1]]
    for i in range(n - 1, -1, -1):
        if np.isnan(x[i]):
            x[i] = nextv
        else:
            nextv = x[i]
    return x


def smooth_1d(x: NDArray[np.floating], win: int = 11) -> NDArray[np.floating]:
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n == 0:
        return x

    # 窗长：奇数且不超过序列长度
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    if win > n:
        win = n if (n % 2 == 1) else max(1, n - 1)
    if win <= 1:
        return x.copy()

    half = win // 2
    k = np.hanning(win)
    k /= (k.sum() + 1e-12)

    # 用边界值/反射做 padding，避免手工“linspace”造成长度不一致
    xpad = np.pad(x, (half, half), mode="edge")   # 如需更平滑也可改 "reflect"
    y = np.convolve(xpad, k, mode="valid")        # 长度与 x 相同
    return y


def zscore_safe(x: NDArray[np.floating]) -> NDArray[np.floating]:
    x = np.asarray(x, dtype=float)
    m, s = np.mean(x), np.std(x) + 1e-8
    return (x - m) / s


def deriv_1d(x: NDArray[np.floating]) -> NDArray[np.floating]:
    d = np.diff(x, prepend=x[:1])
    return d


def cumsum_lowpass(x: NDArray[np.floating], alpha: float = 0.995) -> NDArray[np.floating]:
    # 累积+强低通，生成慢漂移
    y = np.zeros_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc = alpha * acc + (1 - alpha) * v
        y[i] = acc
    return y


def sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    return 1 / (1 + np.exp(-x))


def soft_clip(x: NDArray[np.floating], lim: float = 3.0) -> NDArray[np.floating]:
    # 平滑限幅（|x|>lim 时逐渐压缩）
    return np.tanh(x / lim) * lim / 1.0


def exp_fade_to_zero(x: NDArray[np.floating], lam: float) -> NDArray[np.floating]:
    # 指数平滑向 0（中性位）回归
    y = np.zeros_like(x)
    prev = 0.0
    for i, v in enumerate(x):
        prev = lam * prev + (1 - lam) * v
        y[i] = prev
    return y


def rotvec_from_matrix(R: NDArray[np.floating]) -> NDArray[np.floating]:
    # 将 3x3 旋转矩阵转为轴角向量
    # 参考 Rodrigues：angle = arccos((trace(R)-1)/2)
    tr = np.trace(R)
    cos_theta = (tr - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=float)
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz]) / (2.0 * np.sin(theta) + 1e-12)
    return axis * theta


def R_from_euler_xyz(rx: float, ry: float, rz: float) -> NDArray[np.floating]:
    # Rx * Ry * Rz
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def R_from_euler_zyx(rx: float, ry: float, rz: float) -> NDArray[np.floating]:
    # 按 'zyx' 语义：先绕 x=roll，再绕 y=pitch，再绕 z=yaw 组合为 R = Rz * Ry * Rx
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def euler_to_rotvec_batch(
    roll_x: NDArray[np.floating],
    pitch_y: NDArray[np.floating],
    yaw_z: NDArray[np.floating],
    order: str = "zyx",
) -> NDArray[np.floating]:
    """
    将欧拉角轨迹批量转为轴角 rotvec（每帧一向量）。
    order='zyx'：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  —— 常见的 yaw-pitch-roll。
    order='xyz'：R = Rx(roll) @ Ry(pitch) @ Rz(yaw)。
    """
    T = len(roll_x)
    out = np.zeros((T, 3), dtype=float)
    for i in range(T):
        rx, ry, rz = roll_x[i], pitch_y[i], yaw_z[i]
        if order.lower() == "xyz":
            R = R_from_euler_xyz(rx, ry, rz)
        else:
            R = R_from_euler_zyx(rx, ry, rz)
        out[i] = rotvec_from_matrix(R)
    return out


def ted_presentation_pose(audio, sr=16000, fps=25, num_frames=250):
    """
    High-energy TED Talk style head motion.
    Large, expressive, rhythmic, but not influenced by audio amplitude.
    
    Returns rotvec [T, 3] = [pitch, yaw, roll] in radians.
    """

    T = num_frames

    t = np.linspace(0, 1, T)

    # === Amplitude (radians) ===
    yaw_max   = np.deg2rad(25)   # Big left-right sweep
    pitch_max = np.deg2rad(15)   # Strong nodding
    roll_max  = np.deg2rad(6)    # Small natural roll

    # === 1) Yaw: strong, expressive left-right motion ===
    # Multi-frequency to create stage-performance richness
    yaw = (
        0.70 * np.sin(2 * np.pi * 2.2 * t) +    # main sweep
        0.30 * np.sin(2 * np.pi * 4.8 * t + 0.5)
    ) * yaw_max

    # === 2) Pitch: confident TED-style nodding ===
    pitch = (
        0.85 * np.sin(2 * np.pi * 1.4 * t + 0.3) +
        0.15 * np.sin(2 * np.pi * 3.0 * t + 1.1)
    ) * pitch_max

    # === 3) Roll: subtle tilt for realism ===
    roll = (
        0.50 * np.sin(2 * np.pi * 1.7 * t + 1.0)
    ) * roll_max

    # === output axis-angle vectors ===
    rotvec = np.stack([pitch, yaw, roll], axis=1).astype(np.float32)
    return rotvec