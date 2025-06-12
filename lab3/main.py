import numpy as np
import matplotlib.pyplot as plt

# Функция амплитудного сигнала
def signal_am(amp=1.0, km=0.25, fc=10.0, fs=2.0, period=100):
    t = 2.0 * np.pi * np.linspace(0, 1, period)
    return amp * (1 + km * np.cos(fs * t)) * np.cos(fc * t)

# график АМ сигнала при изменении частоты модулирующего колебания
# в спектре центральная несущая и две боковые моделирующие
N = 1024
fs = 15
fc = [50, 75, 100]
sig = [signal_am(amp=1.0, km=0.45, fc=i, fs=fs, period=N) for i in fc]
sft = np.abs(np.fft.rfft(sig, axis=1)) / N / 0.5
fig = plt.figure(figsize=(12, 6), dpi=80)
for i, freq in enumerate(fc):
    plt.subplot(len(fc), 2, 2*i+1)
    if i == 0:
        plt.title('AM-сигнал')
    plt.plot(sig[i])
    plt.xlim([0, N-1])
    plt.grid(True)

    plt.subplot(len(fc), 2, 2*i+2)
    if i == 0:
        plt.title('Спектр')
    plt.plot(sft[i])
    plt.xlim([0, N//2-1])
    plt.grid(True)
plt.tight_layout()
fig.savefig('AM-сигнал-частота.png')
plt.close(fig)

# график АМ-сигналов при изменении параметра коэффицента модуляции
N = 1024
fs = 6
fc = 40
km = [0.35, 0.85, 5]
sig = [signal_am(amp=1.0, km=i, fc=fc, fs=fs, period=N) for i in km]
sft = np.abs(np.fft.rfft(sig, axis=1)) / N / 0.5
fig = plt.figure(figsize=(12, 6), dpi=80)
for i, freq in enumerate(km):
    plt.subplot(len(km), 2, 2*i+1)
    if i == 0:
        plt.title('AM-сигнал')
    plt.plot(sig[i])
    plt.xlim([0, N-1])
    plt.grid(True)

    plt.subplot(len(km), 2, 2*i+2)
    if i == 0:
        plt.title('Спектр')
    plt.plot(sft[i])
    plt.xlim([0, N//2-1])
    plt.grid(True)
plt.tight_layout()
fig.savefig('AM-сигнал-коэффицент.png')
plt.close(fig)

# Частотная модуляция
def signal_fm(amp=1.0, kd=0.25, fc=10.0, fs=2.0, period=100):
    t = 2.0 * np.pi * np.linspace(0, 1, period)
    return amp * np.cos(fc * t + kd/fs * np.sin(fs * t))

# как моделирующий сигнал, меняет несущий, чтобы получился ЧМ-сигнал
N = 1024
sig = signal_fm(amp=1.0, kd=15, fc=40, fs=4, period=N)
smd = np.cos(4 * 2.0 * np.pi * np.linspace(0, 1, N))
car = np.sin(40 * 2.0 * np.pi * np.linspace(0, 1, N))
fig = plt.figure(figsize=(12, 7), dpi=80)
plt.subplot(2, 1, 1)
plt.plot(smd, label='Модулирующий информационный сигнал')
plt.plot(car, label='Несущий сигнал')
plt.xlim([0, N//2-1])
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Частотно-модулированный сигнал')
plt.plot(sig)
plt.xlim([0, N//2-1])
plt.grid(True)
fig.savefig('Несущий и моделирующий и ЧМ-сигнал.png')
plt.close(fig)

# ЧМ-сигнал в зависимости от значения девиацияя частоты
N = 1024
fs = 5
fc = 65
kd = [5, 25, 40]
sig = [signal_fm(amp=1.0, kd=i, fc=fc, fs=fs, period=N) for i in kd]
sft = np.abs(np.fft.fft(sig, axis=1))
fig = plt.figure(figsize=(12, 6), dpi=80)
for i, freq in enumerate(kd):
    plt.subplot(len(kd), 2, 2*i+1)
    if i == 0:
        plt.title('ЧM-сигнал')
    plt.plot(sig[i])
    plt.xlim([0, N//2-1])
    plt.grid(True)

    plt.subplot(len(kd), 2, 2*i+2)
    if i == 0:
        plt.title('Спектр')
    plt.plot(sft[i])
    plt.xlim([0, N//2-1])
    plt.grid(True)
plt.tight_layout()
fig.savefig('ЧM-сигнал-частота.png')
plt.close(fig)

# Амплитудная манипулляция
N = 64
np.random.seed(1)
mod_rnd = np.random.randint(0, 2, 40)
mod_ask = np.repeat(mod_rnd, repeats=N)
M = mod_ask.size
sig_ask = mod_ask * np.sin(64 * 2.0 * np.pi * np.linspace(0, 1, M))
fig = plt.figure(figsize=(12, 6), dpi=80)
plt.subplot(2, 1, 1)
plt.title('Цифровой сигнал')
plt.plot(mod_ask)
plt.xlim([0, M-1])
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title('АМн-сигнал')
plt.plot(mod_ask, '--b', linewidth='2.0')
plt.plot(sig_ask, '-r')
plt.xlim([0, M-1])
plt.grid(True)
plt.tight_layout()
fig.savefig('АМн-сигнал.png')
plt.close(fig)

# Чистотная манипуляция
N = 100
np.random.seed(1)
mod_rnd = np.random.randint(0, 2, 20)
mod_fsk = np.repeat(mod_rnd, repeats=N)
M = mod_fsk.size
mod_frq = np.zeros(M)
mod_frq[mod_fsk == 0] = 10
mod_frq[mod_fsk == 1] = 50
sig_fsk = np.sin(mod_frq * 2.0 * np.pi * np.linspace(0, 1, M))
fig = plt.figure(figsize=(12, 6), dpi=80)
plt.subplot(2, 1, 1)
plt.title('Цифровой сигнал')
plt.plot(mod_fsk)
plt.xlim([0, M-1])
plt.grid(True)


plt.subplot(2, 1, 2)
plt.title('ЧМн-сигнал')
plt.plot(mod_fsk, '--b', linewidth='2.0')
plt.plot(sig_fsk, '-r')
plt.xlim([0, M-1])
plt.grid(True)
plt.tight_layout()
fig.savefig('ЧМн-сигнал.png')
plt.close(fig)

# Фазовая манипуляция
N = 200
np.random.seed(1)
mod_rnd = np.random.randint(0, 2, 25)
mod_psk = np.repeat(mod_rnd, repeats=N)
M = mod_psk.size
sig_psk = np.sin(25 * 2.0 * np.pi * np.linspace(0, 1, M) + np.pi * mod_psk)
fig = plt.figure(figsize=(12, 6), dpi=80)
plt.subplot(2, 1, 1)
plt.title('Цифровой сигнал')
plt.plot(mod_psk, color='C0', linewidth=2.0)
plt.xlim([0, M-1])
plt.grid(True)


plt.subplot(2, 1, 2)
plt.title('ФМн-сигнал')
plt.plot(mod_psk, '--b', linewidth='2.0')
plt.plot(sig_psk, '-r')
plt.xlim([0, M-1])
plt.grid(True)
plt.tight_layout()
fig.savefig('ФМн-сигнал.png')
plt.close(fig)