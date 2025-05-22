import numpy as np
import matplotlib.pyplot as plt
from task1_2 import load_audio, add_white_noise, plot_time_domain, plot_spectrum
from task4_6 import amplitude_modulation, amplitude_demodulation
import sounddevice as sd
import soundfile as sf

# 设置中文字体和负号显示
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def ideal_bandpass_channel(signal, sr, low_freq, high_freq):
    signal_fft = np.fft.fft(signal)
    n = len(signal_fft)
    
    freq = np.fft.fftfreq(n, 1/sr)
    
    bandpass_filter = np.zeros(n, dtype=complex)
    bandpass_filter[((freq >= low_freq) & (freq <= high_freq)) | 
               ((freq >= -high_freq) & (freq <= -low_freq))] = 1
    
    filtered_fft = signal_fft * bandpass_filter
    filtered_signal = np.real(np.fft.ifft(filtered_fft))
    
    return filtered_signal, bandpass_filter


def transmit_signal(modulated_signal, sr, low_freq, high_freq):
    transmitted_signal, channel_response = ideal_bandpass_channel(modulated_signal, sr, low_freq, high_freq)
    return transmitted_signal, channel_response


def plot_channel_response(channel_response, sr, title, ax=None):
    n = len(channel_response)
    freq = np.fft.fftfreq(n, 1/sr)
    
    idx = np.argsort(freq)
    sorted_freq = freq[idx]
    sorted_response = np.abs(channel_response)[idx]
    
    if ax is None:
        ax = plt.gca()
    ax.plot(sorted_freq, sorted_response)
    ax.set_title(title)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("幅度")
    ax.grid(True)
    ax.set_xlim(-sr/2, sr/2)


if __name__ == "__main__":
    # 读取原始音频
    samples, sr = load_audio("audio/test.m4a")
    
    # 添加白噪声
    noisy_signal, noise = add_white_noise(samples, 15)
    
    # 幅度调制
    modulated_signal, carrier = amplitude_modulation(noisy_signal, sr)
    
    # 设置带通信道参数
    center_freq = sr / 8
    bandwidth = 8000 
    low_freq = center_freq - bandwidth / 2
    high_freq = center_freq + bandwidth / 2
    
    # 通过理想带通信道传输
    transmitted_signal, channel_response = transmit_signal(modulated_signal, sr, low_freq, high_freq)
    
    # 保存传输后的信号
    sf.write("transmitted_signal.wav", transmitted_signal, sr)
    
    plt.figure(figsize=(14, 12), dpi=100)

    # 时域图
    plt.subplot(3, 1, 1)
    plot_time_domain(transmitted_signal, sr, "通过理想带通信道后信号时域图")
    
    # 频谱图
    plt.subplot(3, 1, 2)
    plot_spectrum(transmitted_signal, sr, "通过理想带通信道后信号频谱图")
    
    # 信道频率响应
    ax3 = plt.subplot(3, 1, 3)
    plot_channel_response(channel_response, sr, "理想带通信道频率响应", ax=ax3)
    
    plt.tight_layout()
    plt.savefig("3_5_all_figures.png")
    
    # 播放通过信道后的信号
    print("正在播放通过理想带通信道后的信号...")
    sd.play(transmitted_signal, sr)
    sd.wait()
