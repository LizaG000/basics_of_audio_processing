import numpy as np
import matplotlib.pyplot as plt

# Первая часть: Аналоговый, дискретный и квантовый сигнал
t = np.linspace(0, 1, 50, endpoint=True)
f = 3 * np.cos(np.pi * t) + np.cos(0.5 * np.pi * t) + np.cos(7 * np.pi * t + 5)
titles = ['Аналоговый сигнал', 'Дискретный сигнал', 'Квантовый сигнал']

fig = plt.figure(figsize=(16, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.title(titles[i])
    if i == 0:
        plt.plot(t, f, label="Аналоговый")
    elif i == 1:
        plt.stem(t, f, linefmt='b-', markerfmt='bo', basefmt=" ", label="Дискретный")
    elif i == 2:
        plt.step(t, f, where='mid', label="Квантованный")
    plt.xlim([0, 1])
    plt.yticks(np.linspace(6, -6, 13))
    plt.grid(True)
    plt.legend()
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))

plt.tight_layout()
fig.savefig('analog_discrete_quantized_signal.png')  # Сохранение файла
plt.close(fig)

# Вторая часть: Демонстрация квантования
t = np.linspace(0, 1, 100, endpoint=True)
f = np.cos(2 * np.pi * t)
steps = 6
tt = np.linspace(0, 1, steps, endpoint=True)

fig = plt.figure(figsize=(10, 7))
plt.plot(t, f, '-b', label="Аналоговый сигнал")
plt.step(tt, np.cos(2 * np.pi * tt), 'g', label="Дискретный сиогнал", where='mid')
plt.plot(tt, np.cos(2 * np.pi * tt), '--r', label="Восстановленный аналоговый сигнал")
plt.grid()
plt.xlim([0, 1])
plt.legend()
plt.tight_layout()
fig.savefig('quantization_demo.png')  # Сохранение файла
plt.close(fig)

#Дельта функция
# 
n = 6
t = np.linspace(-n, n-1, 2*n)
fd = np.zeros(2*n)
fd[n] = 1
fig = plt.figure(figsize=(6, 3), dpi=100)
plt.stem(t, fd, markerfmt='D')
plt.xlabel('Отсчеты')
plt.xticks(t)
plt.xlim([np.min(t)+1, np.max(t)])
plt.grid(True)
plt.tight_layout()
fig.savefig('discrete_impulse.png')  # Сохранение файла
plt.close(fig)

# функции Хевисайда
# это функция Хевисайда , она отвечает за включения и выключения, в ЦМОИ используется для включения сигнала в определенный момент времени
n = 6
t = np.linspace(-n, n-1, 2*n)
fh = np.heaviside(t, 1)
fig = plt.figure(figsize=(6, 3), dpi=100)
plt.stem(t, fh, markerfmt='D')
plt.xlabel('Отсчеты')
plt.xticks(t)
plt.xlim([np.min(t)+1, np.max(t)])
plt.grid(True)
plt.tight_layout()
fig.savefig('heaviside_impulse.png')  # Сохранение файла
plt.close(fig)

# Быстрое преобразование Фурье
amplitude = 1.0 # Меняется высота амлитуды
frequency = 100.0 # перемещается скачок
duration = 1.0
sampling_rate = 1000

t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
signal = amplitude*np.sin(2 * np.pi * frequency * t)

fft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), 1 / sampling_rate)

fig = plt.figure(figsize=(15, 7), dpi=100)
plt.plot(frequencies, np.abs(fft))
plt.xlim(0, sampling_rate/2)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Спектр гармонического сигнала')
plt.grid(True)
fig.savefig('spectrum_harmonic_signal.png')  # Сохранение файла
plt.close(fig)

signal1 = np.sin(2 *np.pi * 10 * t )
signal2 = np.sin(2 *np.pi * 100 * t )
signal3 = signal1 + signal2
fft = np.fft.fft(signal1)
frequencies = np.fft.fftfreq(len(signal1), 1 / sampling_rate)
fig = plt.figure(figsize=(15, 7), dpi=100)
plt.plot(frequencies, np.abs(fft))
plt.xlim(0, sampling_rate/2)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Спектр гармонического сигнала')
plt.grid(True)
fig.savefig('spectrum_harmonic_signal1.png')  # Сохранение файла
plt.close(fig)

fft = np.fft.fft(signal2)
frequencies = np.fft.fftfreq(len(signal2), 1 / sampling_rate)
fig = plt.figure(figsize=(15, 7), dpi=100)
plt.plot(frequencies, np.abs(fft))
plt.xlim(0, sampling_rate/2)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Спектр гармонического сигнала')
plt.grid(True)
fig.savefig('spectrum_harmonic_signal2.png')  # Сохранение файла
plt.close(fig)

fft = np.fft.fft(signal3)
frequencies = np.fft.fftfreq(len(signal3), 1 / sampling_rate)
fig = plt.figure(figsize=(15, 7), dpi=100)
plt.plot(frequencies, np.abs(fft))
plt.xlim(0, sampling_rate/2)
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда')
plt.title('Спектр гармонического сигнала')
plt.grid(True)
fig.savefig('spectrum_harmonic_signal3.png')  # Сохранение файла
plt.close(fig)

N = 1024
x = np.zeros(N)
x[128:320] = 1
X = np.fft.fft(x, N)
Xs = np.fft.fftshift(np.abs(X))
f = np.linspace(-0.5, 0.5, N, endpoint=True)

fig = plt.figure(figsize=(8, 8), dpi=100)

plt.subplot(1, 2, 1)
plt.plot(x)
plt.title('Прямоугольный сигнал')
plt.xlabel('Отсчеты')
plt.ylabel('')
plt.xlim([0, N-1])
plt.xticks(np.linspace(0, N, 9, endpoint=True))
plt.grid()
plt.subplot(1, 2, 2)
plt.stem(f, Xs)
plt.title('Амплитудный спектр')
plt.xlabel('Чистота')
plt.ylabel('Амплитуда')
plt.xlim([-1/16, 1/16])
plt.xticks(np.linspace(-1/16, 1/16, 6, endpoint=True))
plt.grid()
plt.tight_layout()
fig.savefig('Амплитудный спектр.png')  # Сохранение файла
plt.close(fig)

harmonics = (4, 16, 32, 128, 256, N//2)
fig = plt.figure(figsize=(14, 9), dpi=100)
for i, j in enumerate(harmonics):
    plt.subplot(3, 2, i+1)
    K = X.copy()
    K[j:] = 0
    k = np.real(np.fft.ifft(K))
    plt.plot(k)
    plt.title(f'Количество гармоник = {j}')
    plt.xlabel('Отсчеты')
    plt.xlim([0, N-1])
    plt.xticks(np.linspace(0, N, 9, endpoint=True))
plt.tight_layout()
fig.savefig('восстановление сигнала.png')  # Сохранение файла
plt.close(fig)