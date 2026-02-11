from pykalman import KalmanFilter
def kalman_1d_numpy(observations, damping=2.0, transition_covariance=0.1):
    """
    Smooth a single 1D sequence using pykalman.
    """
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    kf = KalmanFilter(
        initial_state_mean=initial_value_guess,
        initial_state_covariance=observation_covariance,
        observation_covariance=observation_covariance,
        transition_covariance=transition_covariance,
        transition_matrices=transition_matrix
    )
    pred_state, _ = kf.smooth(observations)
    return pred_state.reshape([-1])

def smooth_with_kalman(latent: torch.Tensor, damping: float = 2.0, transition_covariance: float = 0.1) -> torch.Tensor:
    """
    Apply Kalman smoothing along time axis for each feature dim.

    Args:
        latent: Tensor, shape (B, T, D)
        damping: Observation covariance (the larger, the smoother)
        transition_covariance: Process noise covariance

    Returns:
        smoothed_latent: Tensor, shape (B, T, D)
    """
    if len(latent.shape) ==2:
        latent = latent.unsqueeze(0)
    latent_np = latent.cpu().numpy()
    B, T, D = latent_np.shape

    smoothed_np = np.zeros_like(latent_np)

    for b in range(B):
        for d in range(D):
            smoothed_np[b, :, d] = kalman_1d_numpy(
                latent_np[b, :, d],
                damping=damping,
                transition_covariance=transition_covariance
            )

    smoothed_latent = torch.from_numpy(smoothed_np).to(latent.device).type(latent.dtype)
    return smoothed_latent