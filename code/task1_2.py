import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import warnings
warnings.filterwarnings("ignore")  # 去掉常规警告

def load_audio(file_path):
    # 使用pyhub读取m4a文件
    audio = AudioSegment.from_file(file_path)
    # 将音频转化为numpy数组
    samples = np.array(audio.get_array_of_samples())

    # 如果是立体声，转换为单声道
    if audio.channels == 2:
        samples = samples.reshape((-1, 2))
        samples = samples.mean(axis=1)

    # 获取采样率
    sr = audio.frame_rate

    # 将samples标准化到[-1,1]
    samples = samples / (2**15) if samples.dtype == np.int16 else samples / (2**31)

    return samples, sr

# 绘制时域特征
def plot_time_domain(samples, sr, title):
    duration = len(samples) / sr
    t = np.linspace(-duration // 2, duration // 2, len(samples))

    plt.plot(t, samples)
    plt.title(title)
    plt.xlabel("时间(秒)")
    plt.ylabel('幅度')
    plt.grid()

# 绘制频谱图
def plot_spectrum(samples, sr, title):
    Xw = np.fft.fft(samples)
    w = np.linspace(-sr/2, sr/2, len(Xw))

    plt.plot(w, Xw)
    plt.title(title)
    plt.xlabel("频率w")
    plt.ylabel("频谱")
    plt.grid()

def add_white_noise(signal, snr_db):
    """
    添加高斯白噪声
    signal: 输入信号
    snr_db: 信噪比(dB)
    """
    # 计算信号功率
    signal_power = np.mean(signal**2)
    
    # 根据SNR计算噪声功率
    noise_power = signal_power / (10**(snr_db/10))
    
    # 生成高斯白噪声
    noise = np.random.normal(-np.sqrt(noise_power)/2, np.sqrt(noise_power)/2, len(signal))
    
    # 添加噪声到信号
    noisy_signal = signal + noise
    
    return noisy_signal, noise
 
def add_sinusoidal_noise(signal, sr, freq, amplitude):
    """
    添加正弦噪声
    signal: 输入信号
    sr: 采样率
    freq: 噪声频率(Hz)
    amplitude: 噪声幅度
    """
    # 创建时间向量
    t = np.arange(0, len(signal)) / sr
    
    # 生成正弦噪声
    noise = amplitude * np.sin(2 * np.pi * freq * t)
    
    # 添加噪声到信号
    noisy_signal = signal + noise
    
    return noisy_signal, noise

if __name__ == "__main__":
    audio_file = "signal-system/audio/test.m4a"  
    samples, sr = load_audio(audio_file)
    print(f"样本信号采样率：{sr}, 时长：{len(samples)/sr:.2f}")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    plt.subplots(2, 2, figsize=(14, 8), dpi=100)
    plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
    
    # 绘制时域特征
    plt.subplot(221)
    plot_time_domain(samples, sr, "语音信号波形图")
    
    # 绘制频域图
    plt.subplot(222)
    plot_spectrum(samples, sr, "原始信号频谱图")

    # 添加高斯白噪声
    noisy_signal, noise = add_white_noise(samples, 15)

    # 添加正弦噪声(可选)
    # noisy_signal, noise = add_sinusoidal_noise(samples, sr, 1000, 0.2)

    plt.subplot(223)
    plot_time_domain(noisy_signal, sr, "添加噪声后的信号波形图")
    plt.subplot(224)
    plot_spectrum(noisy_signal, sr, "添加噪声后的信号频谱图")

    plt.savefig("原始信号与噪声信号.png")
    plt.show()

