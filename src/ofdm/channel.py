import numpy as np


def add_awgn(signal, snr_db, rng=None):
    """Add complex additive white Gaussian noise."""
    signal = np.asarray(signal, dtype=complex)

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(np.abs(signal) ** 2)

    if signal_power == 0:
        raise ValueError("Signal must have nonzero power.")

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = np.sqrt(noise_power / 2) * (
        rng.standard_normal(signal.shape)
        + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def apply_multipath(signal, channel_taps=None):
    """Apply a simple multipath channel using linear convolution."""
    signal = np.asarray(signal, dtype=complex)

    if channel_taps is None:
        channel_taps = np.array(
            [1.0, 0.0, 0.35, 0.0, 0.15],
            dtype=complex,
        )

    channel_taps = np.asarray(channel_taps, dtype=complex)

    if signal.size == 0:
        raise ValueError("Signal cannot be empty.")

    if channel_taps.ndim != 1 or channel_taps.size == 0:
        raise ValueError("Channel taps must be a nonempty 1D array.")

    return np.convolve(signal, channel_taps, mode="full")


def channel_frequency_response(channel_taps, fft_size):
    """Calculate the channel frequency response on OFDM subcarriers."""
    return np.fft.fft(channel_taps, n=fft_size)
