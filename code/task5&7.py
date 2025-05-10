import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from task2&3 import load_audio, add_white_noise, plot_time_domain, plot_spectrum
baseband_max_freq = 4000 # Hz 语音信号的频带的最大值

# --- Helper function for low-pass filter ---
def _butter_lowpass_filter(data, cutoff_freq, sample_rate, order=5):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low')
    filtered_data = lfilter(b, a, data)
    return filtered_data

# --- Step 5: Amplitude Modulation ---
def amplitude_modulation(signal, sr):
    t = np.arange(len(signal)) / sr
    fc = sr / 4  # carrier frequency
    carrier = np.cos(2 * np.pi * fc * t) # cos(2πf_ct)
    return signal * carrier, carrier

# --- Step 7: Amplitude Demodulation ---
def amplitude_demodulation(signal, sr, carrier):
    demodulated_signal_raw = signal * carrier
    lpf_cutoff = baseband_max_freq + 500
    demodulated_signal_filtered = _butter_lowpass_filter(demodulated_signal_raw, lpf_cutoff, sr, order=6)
    return demodulated_signal_filtered

if __name__ == "__main__":
    samples, sr = load_audio('/signal-system/audio/test.m4a')
    noisy_signal, noise = add_white_noise(samples, 15)
    modulated_signal, carrier = amplitude_modulation(noisy_signal, sr)
    demodulated_signal = amplitude_demodulation(modulated_signal, sr, carrier)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    plt.subplot(221)
    plot_time_domain(modulated_signal, sr, "调制后的信号波形图")
    plt.subplot(222)
    plot_spectrum(modulated_signal, sr, "调制后的信号频谱图")
    plt.subplot(223)
    plot_time_domain(demodulated_signal, sr, "解调后的信号波形图")
    plt.subplot(224)
    plot_spectrum(demodulated_signal, sr, "解调后的信号频谱图")
    plt.savefig('5&7.png')