import torch
from pytorch3d.renderer import look_at_view_transform


def space_orbit_camera(num_frames=120, radius=1.0, elevation=10.0):
    """
    Generate camera poses (R, T) that orbit around the origin in 3D.

    Args:
        num_frames (int): number of frames
        radius (float): distance from the origin
        elevation (float): vertical angle in degrees

    Returns:
        R: [N, 3, 3] camera rotation matrices
        T: [N, 3] camera translation vectors
    """

    azim = torch.linspace(0, 360, num_frames)
    elev = torch.full_like(azim, elevation)

    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)
    return R, T


def singer_mv_orbit_zoom(num_frames=120, radius=1.0, elevation=10.0):
    radius_range=(1.2, 0.9)

    t = torch.linspace(0, 1, num_frames)


    azim = 45 * torch.sin(2 * torch.pi * t)  


    elev = torch.full_like(azim, elevation) + 2.0 * torch.sin(2 * torch.pi * t)


    half = num_frames //2
    radius = torch.linspace(radius_range[0], radius_range[1], half)
    radius_back = torch.linspace(radius_range[1], radius_range[0], num_frames-half)
    radius = torch.cat([radius, radius_back])

    # from pytorch3d.renderer import look_at_view_transform
    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)

    return R, T



def beat_drop_bounce(num_frames=120, radius=1.1, elevation=10.0, beat_freq=4):
    t = torch.linspace(0, 1, num_frames)
    beat = torch.sin(2 * torch.pi * beat_freq * t)


    azim = 10.0 * torch.sin(2 * torch.pi * t * beat_freq / 2)


    elev = torch.full_like(t, elevation) + 3.0 * torch.sin(4 * torch.pi * t * beat_freq)

    new_radius = radius + 0.2 * beat  # beat ∈ [-1, 1]


    from pytorch3d.renderer import look_at_view_transform
    R, T = look_at_view_transform(dist=new_radius, elev=elev, azim=azim)

    return R, T


def ted_talk_camera_energetic(num_frames=120, radius=1.2, elevation=11.0):

    t = torch.linspace(0, 1, num_frames)

    azim = 15.0 * torch.sin(3 * torch.pi * t)


    elev = torch.full_like(azim, elevation) + 1.2 * torch.sin(2 * torch.pi * t)


    radius_start, radius_mid, radius_end = radius, radius * 0.8, radius * 1.0
    radius_curve = (
        radius_start
        - (radius_start - radius_mid) * torch.sin(torch.pi * t) ** 2
        + 0.05 * torch.sin(4 * torch.pi * t)  
    )

    R, T = look_at_view_transform(dist=radius_curve, elev=elev, azim=azim)

    return R, T



def camera_orbit_parallax(num_frames=120, radius=1.6, elevation=10.0,
                          arc_deg=60.0, seed: int = 0):
    torch.manual_seed(seed)
    t = torch.linspace(0, 1, num_frames)
    s = 0.5 - 0.5 * torch.cos(torch.pi * t) 


    azim = -arc_deg / 2 + arc_deg * s


    dist = radius * (1.0 + 0.15 * torch.cos(2 * torch.pi * s))


    elev = torch.full_like(azim, elevation) + 1.2 * torch.sin(2 * torch.pi * s + torch.pi/4)


    azim = azim + 0.6 * torch.randn_like(azim)
    elev = elev + 0.25 * torch.randn_like(elev)
    dist = dist + 0.02 * torch.randn_like(dist)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return R, T


def _ease_in_out_sine(t: torch.Tensor) -> torch.Tensor:
    return 0.5 - 0.5 * torch.cos(torch.pi * t)

def _ease_out_expo(t: torch.Tensor) -> torch.Tensor:
    return 1 - torch.exp(-12 * torch.clamp(t, 0, 1))

def camera_push_then_whip_pull(
    num_frames: int = 120,
    radius: float = 1.5,
    elevation: float = 9.0,
    push_ratio: float = 0.70,     
    push_amount: float = 0.12,    
    whip_amount: float = 0.40,   
    whip_kick_deg: float = 6.0,   
    arc_deg: float = 4.0,        
    settle_osc: float = 0.06,     
):

    t = torch.linspace(0, 1, num_frames)
    tw = torch.tensor(push_ratio)


    s_push = _ease_in_out_sine(torch.clamp(t / tw, 0, 1))
    dist_push = radius * (1.0 - push_amount * s_push)
    dist_at_whip = radius * (1.0 - push_amount)


    u = torch.clamp((t - tw) / (1 - tw), 0, 1) 
    pull_main = _ease_out_expo(u)                              
    pull_osc  = settle_osc * torch.sin(2 * torch.pi * u) * torch.exp(-4 * u)  
    dist_pull = dist_at_whip + radius * (whip_amount * pull_main + pull_osc)

    dist = torch.where(t <= tw, dist_push, dist_pull)


    base_azim = arc_deg * torch.sin(2 * torch.pi * t) 
   
    sigma = 0.035
    whip_pulse = torch.exp(-0.5 * ((t - tw) / sigma) ** 2)   
    azim = base_azim - whip_kick_deg * whip_pulse             

    elev = torch.full_like(azim, elevation)
    elev = elev + 0.6 * torch.sin(2 * torch.pi * t + torch.pi / 4) \
                 + (-0.25 * whip_kick_deg) * whip_pulse / max(whip_kick_deg, 1e-6)

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return R, T


def _ease_in_out_sine(t: torch.Tensor) -> torch.Tensor:
    return 0.5 - 0.5 * torch.cos(torch.pi * torch.clamp(t, 0, 1))

def camera_low_angle_hero(
    num_frames: int = 250,
    radius: float = 1.55,         
    elevation: float = -14.0,    
    azim_start: float = -20.0,    
    azim_end: float = -4.0,      
    push_in: float = 0.3,       
    hold_ratio: float = 0.10,    
    micro_breathe: float = 0.6,  
):

    t = torch.linspace(0, 1, num_frames)
    move_ratio = 1.0 - hold_ratio


    s = torch.where(
        t <= move_ratio,
        _ease_in_out_sine(t / move_ratio),
        torch.ones_like(t)
    )

    azim = azim_start + (azim_end - azim_start) * s

    dist = radius * (1.0 - push_in * s)
    decay = torch.exp(-4.0 * s)   # s→1 时 -> 0
    elev = torch.full_like(azim, elevation) + micro_breathe * torch.sin(2 * torch.pi * t) * decay

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    return R, T




def coco_pixar_emotional_arc(num_frames=120, base_radius=1.25, base_elev=6.0):
    """
    Pixar / Coco cinematic camera:
    - warm emotional arcs
    - expressive push-in
    - soft breathing elevation
    - heartfelt roll tilt
    """

    t = torch.linspace(0, 1, num_frames)

    # 1) Emotional push-in (0.30 units) – much stronger than previous versions
    push = 0.30 * torch.sin(torch.pi * t)
    radius = base_radius - push

    # 2) Pixar signature warm arc sweep (±18°)
    # Large, graceful horizontal sweep
    azim = 18.0 * torch.sin(1.1 * torch.pi * t)

    # 3) Gentle breathing elevation (±4.5°)
    elev = base_elev + 4.5 * torch.sin(1.7 * torch.pi * t + 0.3)

    # 4) Emotional roll (±5°) – subtle but emotional
    roll = 5.0 * torch.sin(2.2 * torch.pi * t + 0.5)

    # generate R,T
    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)

    return R, T


def broadway_swing_camera(num_frames=120, base_radius=1.35, base_elev=15.0):
    """
    Broadway Swing-style cinematic camera motion:
    - large lateral swings (stage sweep)
    - rhythmic push-in pulses
    - elevated stage viewpoint
    - expressive roll swing
    """

    t = torch.linspace(0, 1, num_frames)

    # 1) Strong rhythmic push-in (0.20 units)
    push = 0.20 * torch.sin(2.0 * torch.pi * t)   # twice the normal beat
    radius = base_radius - push

    # 2) Broadway-style stage sweep (±25°)
    # Strong sideways swinging like sweeping spotlights
    azim = 25.0 * torch.sin(1.3 * torch.pi * t)

    # 3) High-angle stage elevation (±6° breathing)
    elev = base_elev + 6.0 * torch.sin(0.9 * torch.pi * t + 0.6)

    # 4) Dramatic roll swing (±8°)
    roll = 8.0 * torch.sin(2.5 * torch.pi * t + 0.2)

    R, T = look_at_view_transform(dist=radius, elev=elev, azim=azim)

    return R, T



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import splprep, splev
    
    # def camera_spiral_path(r_start=2.0, r_end=4.0, num_frames=180, turns=1.5):
    #     theta = np.linspace(0, 2 * np.pi * turns, num_frames)
    #     r = np.linspace(r_start, r_end, num_frames)
    #     x = r * np.cos(theta)
    #     y = r * np.sin(theta)
    #     return x, y

    # x, y = camera_spiral_path()
    # plt.plot(x, y, lw=2, color='tomato')
    # plt.axis('equal')
    # plt.title('Spiral Camera Path (XY Plane)')
    # plt.xlabel('X'); plt.ylabel('Y')
    # plt.grid(ls='--', alpha=0.4)
    # plt.show()

    # from scipy.interpolate import splprep, splev

    def smooth_random_path(num_points=6, num_frames=250, scale=2.0, seed=42):
        np.random.seed(seed)
        ctrl_x = np.random.uniform(-scale, scale, num_points)
        ctrl_y = np.random.uniform(-scale, scale, num_points)
        tck, u = splprep([ctrl_x, ctrl_y], s=2.0)
        u_new = np.linspace(0, 1, num_frames)
        x, y = splev(u_new, tck)
        return x[::-1], y[::-1]

    x, y = smooth_random_path()


    plt.figure(figsize=(6, 5))
    plt.plot(x, y, color='gray', lw=2, alpha=0.9)


    skip = 10
    for i in range(0, len(x) - skip, skip):
        dx, dy = x[i + skip] - x[i], y[i + skip] - y[i]
        plt.arrow(
            x[i], y[i], dx, dy,
            shape='full', lw=0, length_includes_head=True,
            head_width=0.1, color='gray', alpha=0.6
        )

    plt.scatter(x[0], y[0], color='green', s=60, label='Start')
    plt.scatter(x[-1], y[-1], color='red', s=60, label='End')
    # plt.text(x[0], y[0] + 0.15, "Start", fontsize=9, color='green', ha='center')
    # plt.text(x[-1], y[-1] + 0.15, "End", fontsize=9, color='red', ha='center')

    plt.axis('equal')
    plt.grid(ls='--', alpha=0.4)
    plt.xlabel("X"); plt.ylabel("Y")
    plt.title("Smooth Random Camera Path (XY Plane, Corrected Direction)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()