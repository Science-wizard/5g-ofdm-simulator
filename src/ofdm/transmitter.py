import numpy as np


def generate_bits(num_bits: int) -> np.ndarray:
    """Generate random binary information."""
    return np.random.randint(0, 2, num_bits)


def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """Map pairs of bits to normalized QPSK symbols."""
    if len(bits) % 2 != 0:
        raise ValueError("QPSK requires an even number of bits.")

    pairs = bits.reshape(-1, 2)
    real = 1 - 2 * pairs[:, 0]
    imag = 1 - 2 * pairs[:, 1]

    return (real + 1j * imag) / np.sqrt(2)


def ofdm_modulate(
    symbols: np.ndarray,
    fft_size: int,
    cp_length: int,
) -> np.ndarray:
    """Generate an OFDM waveform with a cyclic prefix."""
    if len(symbols) != fft_size:
        raise ValueError("Number of symbols must equal FFT size.")

    if not 0 <= cp_length <= fft_size:
        raise ValueError("Invalid cyclic prefix length.")

    time_signal = np.fft.ifft(symbols, n=fft_size)
    prefix = time_signal[-cp_length:] if cp_length else np.array([], dtype=complex)

    return np.concatenate((prefix, time_signal))


def generate_ofdm_signal(
    fft_size: int = 64,
    cp_length: int = 16,
) -> dict:
    """Generate one simplified OFDM transmission."""
    bits = generate_bits(2 * fft_size)
    symbols = qpsk_modulate(bits)
    signal = ofdm_modulate(symbols, fft_size, cp_length)

    return {
        "bits": bits,
        "symbols": symbols,
        "signal": signal,
        "fft_size": fft_size,
        "cp_length": cp_length,
    }
