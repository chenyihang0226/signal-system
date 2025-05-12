import numpy as np
import matplotlib.pyplot as plt
from task1_2 import load_audio, add_white_noise, plot_time_domain, plot_spectrum
import sounddevice as sd
import soundfile as sf

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def wiener_filter(signal, noise_power=None):
    # 信号转换为频域
    signal_fft = np.fft.fft(signal)
    signal_power_spectrum = np.abs(signal_fft) ** 2

    if noise_power is not None:
        noise_power_spectrum = np.ones_like(signal_fft) * noise_power
    else:
        # 如果没有提供噪声信息，尝试从信号中估计噪声功率
        noise_power_spectrum = np.mean(
            signal_power_spectrum[-int(len(signal) / 10) :]
        ) * np.ones_like(signal_fft)

    # 维纳滤波器传递函数：H(f) = Ps(f) / (Ps(f) + Pn(f))
    wiener_filter = signal_power_spectrum / (
        signal_power_spectrum + noise_power_spectrum
    )

    # 应用滤波器
    filtered_fft = signal_fft * wiener_filter

    # 转回时域
    filtered_signal = np.real(np.fft.ifft(filtered_fft))

    return filtered_signal


if __name__ == "__main__":
    # 读取原始音频
    samples, sr = load_audio("audio/test.m4a")
    # 添加白噪声
    noisy_signal, noise = add_white_noise(samples, 15)
    # 滤波
    filtered_signal = wiener_filter(noisy_signal, noise_power=7.5)
    # 保存加噪声后的信号
    sf.write("noisy_signal.wav", noisy_signal, sr)
    # 保存滤波后的信号
    sf.write("filtered_signal.wav", filtered_signal, sr)

    # 绘图
    plt.figure(figsize=(14, 8), dpi=100)
    plt.subplot(221)
    plot_time_domain(noisy_signal, sr, "加噪声信号时域图")
    plt.subplot(222)
    plot_spectrum(noisy_signal, sr, "加噪声信号频谱图")
    plt.subplot(223)
    plot_time_domain(filtered_signal, sr, "滤波后信号时域图")
    plt.subplot(224)
    plot_spectrum(filtered_signal, sr, "滤波后信号频谱图")
    plt.tight_layout()
    plt.savefig("8&9.png")

    # 播放原始、加噪声、滤波后信号
    print("正在播放原始信号...")
    sd.play(samples, sr)
    sd.wait()
    print("正在播放加噪声信号...")
    sd.play(noisy_signal, sr)
    sd.wait()
    print("正在播放滤波后信号...")
    sd.play(filtered_signal, sr)
    sd.wait()
