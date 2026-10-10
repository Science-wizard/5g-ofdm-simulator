import numpy as np


def ofdm_demodulate(signal, fft_size, cp_length):
    """Remove the cyclic prefix and recover frequency-domain symbols."""
    signal = np.asarray(signal, dtype=complex)

    if not 0 <= cp_length < fft_size:
        raise ValueError("Cyclic prefix must be shorter than FFT size.")

    symbol_length = fft_size + cp_length

    if signal.size < symbol_length:
        raise ValueError("Received signal is shorter than one OFDM symbol.")

    useful = signal[cp_length:symbol_length]
    return np.fft.fft(useful, n=fft_size)


def qpsk_demodulate(symbols):
    """Recover bits from normalized QPSK symbols."""
    bits = np.empty(2 * len(symbols), dtype=int)
    bits[0::2] = (np.real(symbols) < 0).astype(int)
    bits[1::2] = (np.imag(symbols) < 0).astype(int)
    return bits


def equalize(received_symbols, channel_response):
    """Perform zero-forcing frequency-domain equalization."""
    received_symbols = np.asarray(received_symbols, dtype=complex)
    channel_response = np.asarray(channel_response, dtype=complex)

    if np.any(np.abs(channel_response) < 1e-12):
        raise ValueError("Channel response contains a deep spectral null.")

    return received_symbols / channel_response
