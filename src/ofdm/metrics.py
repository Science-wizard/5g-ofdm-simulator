import numpy as np


def bit_error_rate(transmitted_bits, received_bits):
    """Calculate the fraction of incorrectly recovered bits."""
    tx = np.asarray(transmitted_bits)
    rx = np.asarray(received_bits)

    if tx.shape != rx.shape:
        raise ValueError("Bit arrays must have the same shape.")

    if tx.size == 0:
        raise ValueError("Bit arrays cannot be empty.")

    errors = int(np.count_nonzero(tx != rx))

    return {
        "errors": errors,
        "total_bits": int(tx.size),
        "ber": errors / tx.size,
    }
