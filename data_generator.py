import numpy as np
import os
from tqdm import tqdm
import config


def generate_ula_data(num_sources, thetas_deg, snr_db, M, L, d_lambda):
    """
    Generates simulated Uniform Linear Array (ULA) data.

    Args:
        num_sources (int): Number of signal sources.
        thetas_deg (list): List of angles of arrival in degrees.
        snr_db (float): Signal-to-Noise Ratio in dB.
        M (int): Number of array elements.
        L (int): Number of snapshots.
        d_lambda (float): Element spacing in wavelengths.

    Returns:
        tuple: A tuple containing the covariance matrix (M, M) and the true angles.
    """
    thetas_rad = np.deg2rad(thetas_deg)

    # Steering matrix
    A = np.exp(1j * 2 * np.pi * d_lambda * np.arange(M)[:, np.newaxis] * np.sin(thetas_rad))

    # Source signals (uncorrelated, complex Gaussian)
    S = (np.random.randn(num_sources, L) + 1j * np.random.randn(num_sources, L)) / np.sqrt(2)

    # Received signal (noise-free)
    Y_clean = A @ S

    # Noise
    signal_power = np.mean(np.abs(Y_clean) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) * np.sqrt(noise_power / 2)

    # Received signal with noise
    Y = Y_clean + noise

    # Covariance matrix
    R = (Y @ Y.conj().T) / L

    return R, thetas_deg


def create_dataset(num_samples, M, L, d_lambda, angle_min, angle_max, grid_size):
    """Creates a dataset with random sources, angles, and SNRs."""
    X = np.zeros((num_samples, M, M), dtype=np.complex64)
    # Use a padded list for labels since the number of sources varies
    Y = -1 * np.ones((num_samples, config.MAX_SOURCES), dtype=np.float32)

    for i in tqdm(range(num_samples)):
        num_sources = np.random.randint(1, config.MAX_SOURCES + 1)

        # Ensure minimum separation between sources
        while True:
            thetas = np.sort(np.random.uniform(angle_min, angle_max, num_sources))
            if num_sources == 1 or np.all(np.diff(thetas) >= 3.0):
                break

        snr_db = np.random.uniform(20, 40)

        R, true_thetas = generate_ula_data(
            num_sources=num_sources,
            thetas_deg=thetas,
            snr_db=snr_db,
            M=M,
            L=L,
            d_lambda=d_lambda
        )

        X[i, :, :] = R
        # Convert angles to grid indices
        angle_indices = np.round((true_thetas - angle_min) / config.ANGLE_RESOLUTION).astype(int)
        Y[i, :num_sources] = angle_indices

    return X, Y


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    print("Generating training data...")
    X_train, Y_train = create_dataset(
        num_samples=config.NUM_TRAIN_SAMPLES,
        M=config.NUM_ARRAY_ELEMENTS,
        L=config.SNAPSHOTS,
        d_lambda=config.ARRAY_SPACING,
        angle_min=config.ANGLE_MIN,
        angle_max=config.ANGLE_MAX,
        grid_size=config.GRID_SIZE
    )
    np.savez(config.TRAIN_DATA_PATH, X=X_train, Y=Y_train)
    print(f"Training data saved to {config.TRAIN_DATA_PATH}")

    print("Generating validation data...")
    X_val, Y_val = create_dataset(
        num_samples=config.NUM_VAL_SAMPLES,
        M=config.NUM_ARRAY_ELEMENTS,
        L=config.SNAPSHOTS,
        d_lambda=config.ARRAY_SPACING,
        angle_min=config.ANGLE_MIN,
        angle_max=config.ANGLE_MAX,
        grid_size=config.GRID_SIZE
    )
    np.savez(config.VAL_DATA_PATH, X=X_val, Y=Y_val)
    print(f"Validation data saved to {config.VAL_DATA_PATH}")