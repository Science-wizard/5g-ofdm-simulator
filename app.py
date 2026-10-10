import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from src.ofdm.transmitter import generate_ofdm_signal
from src.ofdm.channel import add_awgn, apply_multipath, channel_frequency_response
from src.ofdm.receiver import ofdm_demodulate, qpsk_demodulate, equalize
from src.ofdm.metrics import bit_error_rate

st.set_page_config(
    page_title="5G OFDM Signal Simulator",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
.stApp {
    background-color: #0B0F19;
    color: #E5E7EB;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
div[data-testid="stMetric"] {
    background-color: #111827;
    padding: 16px;
    border-radius: 12px;
    border: 1px solid #263244;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📡 5G LAB")
st.sidebar.caption("Wireless Communications Laboratory")

screen = st.sidebar.radio(
    "NAVIGATION",
    ["Overview", "Transmitter", "Channel", "Receiver", "Performance"],
)

st.title("5G OFDM Signal Simulator")
st.caption("An interactive laboratory for digital wireless communications")
st.divider()

if "ofdm_result" not in st.session_state:
    st.session_state["ofdm_result"] = None

if "channel_result" not in st.session_state:
    st.session_state["channel_result"] = None

if screen == "Overview":
    st.subheader("Simulation Overview")

    a, b, c = st.columns(3)
    a.metric("FFT Size", "64")
    b.metric("Modulation", "QPSK")
    c.metric("System Status", "Ready")

    st.markdown("### Communication pipeline")
    st.code(
        "Bits → QPSK → IFFT → Cyclic Prefix → Channel\n"
        "    → Remove CP → FFT → Equalize → Recover Bits → BER"
    )

    st.info("Generate a signal in Transmitter to begin your experiment.")

elif screen == "Transmitter":
    st.subheader("OFDM Transmitter")
    st.write("Convert random binary data into a time-domain OFDM waveform.")

    fft_size = st.selectbox(
        "Number of subcarriers",
        [64, 128, 256, 512, 1024],
        key="tx_fft",
    )

    cp_length = st.selectbox(
        "Cyclic prefix length",
        [4, 8, 16, 32],
        index=2,
        key="tx_cp",
    )

    if st.button("Generate OFDM Signal", type="primary"):
        st.session_state["ofdm_result"] = generate_ofdm_signal(
            fft_size=fft_size,
            cp_length=cp_length,
        )
        st.session_state["channel_result"] = None

    result = st.session_state["ofdm_result"]

    if result is not None:
        signal = result["signal"]
        symbols = result["symbols"]

        a, b, c = st.columns(3)
        a.metric("Subcarriers", result["fft_size"])
        b.metric("OFDM Samples", len(signal))
        c.metric("Cyclic Prefix", result["cp_length"])

        st.markdown("### Binary information")
        st.code("".join(map(str, result["bits"][:64])))

        st.markdown("### QPSK constellation")
        fig, ax = plt.subplots()
        ax.scatter(symbols.real, symbols.imag)
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Time-domain waveform")
        fig, ax = plt.subplots()
        ax.plot(signal.real, label="In-phase")
        ax.plot(signal.imag, label="Quadrature")
        ax.axvline(
            result["cp_length"] - 0.5,
            linestyle="--",
            label="Useful symbol begins",
        )
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("### Frequency-domain subcarriers")
        spectrum = np.fft.fftshift(
            np.fft.fft(signal[result["cp_length"]:])
        )
        frequencies = np.fft.fftshift(
            np.fft.fftfreq(result["fft_size"])
        )

        fig, ax = plt.subplots()
        ax.stem(frequencies, np.abs(spectrum))
        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Magnitude")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

elif screen == "Channel":
    st.subheader("Wireless Channel Laboratory")
    st.write("Introduce noise or multipath propagation into the signal.")

    result = st.session_state["ofdm_result"]

    if result is None:
        st.warning("Generate a signal on the Transmitter screen first.")
    else:
        snr_db = st.slider(
            "Signal-to-Noise Ratio (dB)",
            0, 40, 20,
            key="channel_snr",
        )

        model = st.selectbox(
            "Channel model",
            ["AWGN", "Multipath + AWGN"],
        )

        if st.button("Transmit Through Channel", type="primary"):
            rng = np.random.default_rng(42)
            tx = result["signal"]

            if model == "AWGN":
                channel_taps = np.array([1.0 + 0j])
            else:
                channel_taps = np.array(
                    [1.0, 0.0, 0.35, 0.0, 0.15],
                    dtype=complex,
                )

            faded = np.convolve(tx, channel_taps)
            received = add_awgn(faded, snr_db, rng)

            st.session_state["channel_result"] = {
                "received": received,
                "snr_db": snr_db,
                "model": model,
                "channel_taps": channel_taps,
            }

        channel = st.session_state["channel_result"]

        if channel is not None:
            a, b, c = st.columns(3)
            a.metric("Channel", channel["model"])
            b.metric("SNR", f"{channel['snr_db']} dB")
            c.metric("Received Samples", len(channel["received"]))

            n = min(100, len(result["signal"]), len(channel["received"]))

            fig, ax = plt.subplots()
            ax.plot(result["signal"][:n].real, label="Transmitted I")
            ax.plot(channel["received"][:n].real, label="Received I")
            ax.set_xlabel("Sample index")
            ax.set_ylabel("Amplitude")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "The multipath model adds delayed copies. "
                "The receiver will compensate for this channel in the next stage."
            )

elif screen == "Receiver":
    st.subheader("OFDM Receiver")

    result = st.session_state["ofdm_result"]
    channel = st.session_state["channel_result"]

    if result is None or channel is None:
        st.warning("Generate a signal, then transmit it through the Channel screen.")
    else:
        if st.button("Recover Transmitted Bits", type="primary"):
            try:
                rx = channel["received"]
                cp = result["cp_length"]
                nfft = result["fft_size"]

                # The current channel starts at time zero.
                # Use the first complete OFDM symbol.
                rx_symbols = ofdm_demodulate(rx, nfft, cp)

                h = channel_frequency_response(
                    channel["channel_taps"],
                    nfft,
                )

                equalized = equalize(rx_symbols, h)
                recovered = qpsk_demodulate(equalized)
                metrics = bit_error_rate(result["bits"], recovered)

                st.session_state["receiver_result"] = {
                    "symbols": rx_symbols,
                    "equalized": equalized,
                    "bits": recovered,
                    "metrics": metrics,
                }

            except ValueError as error:
                st.error(str(error))

        receiver = st.session_state.get("receiver_result")

        if receiver is not None:
            metrics = receiver["metrics"]

            a, b, c = st.columns(3)
            a.metric("Recovered Bits", metrics["total_bits"])
            b.metric("Bit Errors", metrics["errors"])
            c.metric("BER", f"{metrics['ber']:.6f}")

            st.markdown("### Received constellation")
            fig, ax = plt.subplots()
            ax.scatter(
                receiver["equalized"].real,
                receiver["equalized"].imag,
                alpha=0.7,
            )
            ax.set_xlabel("In-phase (I)")
            ax.set_ylabel("Quadrature (Q)")
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("### Bit comparison")
            comparison = np.column_stack(
                (result["bits"][:32], receiver["bits"][:32])
            )
            st.dataframe(
                {
                    "Index": np.arange(32),
                    "Transmitted": comparison[:, 0],
                    "Received": comparison[:, 1],
                    "Correct": comparison[:, 0] == comparison[:, 1],
                },
                use_container_width=True,
            )

elif screen == "Performance":
    st.subheader("Performance Analysis")
    st.write("Measure bit errors as the signal-to-noise ratio changes.")

    result = st.session_state["ofdm_result"]

    if result is None:
        st.warning("Generate a signal in the Transmitter screen first.")
    else:
        if st.button("Run BER vs SNR Experiment", type="primary"):
            snr_values = np.arange(0, 31, 5)
            ber_values = []

            for snr in snr_values:
                rng = np.random.default_rng(1000 + int(snr))
                received = add_awgn(result["signal"], snr, rng)

                rx_symbols = ofdm_demodulate(
                    received,
                    result["fft_size"],
                    result["cp_length"],
                )

                rx_bits = qpsk_demodulate(rx_symbols)
                metrics = bit_error_rate(result["bits"], rx_bits)
                ber_values.append(metrics["ber"])

            st.session_state["ber_curve"] = (
                snr_values,
                np.array(ber_values),
            )

        curve = st.session_state.get("ber_curve")

        if curve is not None:
            snr_values, ber_values = curve

            fig, ax = plt.subplots()
            ax.semilogy(
                snr_values,
                np.maximum(ber_values, 1e-5),
                marker="o",
            )
            ax.set_xlabel("SNR (dB)")
            ax.set_ylabel("Bit Error Rate")
            ax.set_title("BER vs SNR — QPSK over AWGN")
            ax.grid(True, which="both")
            st.pyplot(fig)
            plt.close(fig)

            st.dataframe(
                {"SNR (dB)": snr_values, "BER": ber_values},
                use_container_width=True,
            )
