import numpy as np
import matplotlib.pyplot as plt

# Корреляция
x1 = np.array([1, 2, 3, 4, 5 ,6 ,7])
print(f"Корреляция: {np.correlate(x1, x1)}")
print(f"Корреляция: {np.sum(x1*x1)}")
print()

# Отрицательная корреляция
x1 = np.array([1, 2, 3, 4])
x2 = np.array([-1, -2, -3, -4])
print(f"Отрицательная корреляция: {np.correlate(x1, x2)}")
print(f"Отрицательная корреляция: {np.sum(x1*x2)}")
print()

# Случайные процесы с нулевой корреляцией
N = 20
np.random.seed(1)
x1 = np.random.randint(-10, 10, N)
x2 = np.random.randint(-10, 10, N)
r12 = np.correlate(x1, x2)
fig = plt.figure(figsize=(14, 4), dpi=80)
plt.plot(x1, '-d')
plt.plot(x2, '-p')
plt.xlim([-0.5, N-0.5])
plt.grid(True)
print(f'r12 = {r12}')
fig.savefig('Случайные процесы с нулевой корреляцией.png')  # Сохранение файла
plt.close(fig)
print()

# Случайные процесы и дельта функция
N=10
fd = np.zeros(N)
fd[5] = 1
np.random.seed(2)
x = np.random.randn(N)
print(x)
r = np.correlate(fd, x)
print(r)
fig = plt.figure(figsize=(14, 4), dpi=80)
plt.plot(fd, '-o')
plt.plot(x, '-s')
plt.xlim([-0.5, N-0.5])
plt.grid(True)
print(f'r = {r} x[5] = {x[5]}')
fig.savefig('Случайные процесы и дельта функция.png')  # Сохранение файла
plt.close(fig)
print()

# линейнеая свертка
N, M = 4, 3
a = [1, 2, 3, 4]
b = [3, 2 ,1]
an = np.concatenate([np.zeros(M-1, dtype=int), a])
print(an)
bn = np.concatenate([b[::-1], np.zeros(N-1, dtype=int)])
print(bn)
ab = []
for i in range(N+M-1):
    br = np.roll(bn, i)
    sm = np.sum(an * br)
    print(br)
    print(sm)
    ab.append(int(sm))
print(ab)
# вычесление свертки
cv = np.convolve(a, b, mode = 'full')
print(cv)
ab_check = np.all(ab == cv)
print(ab_check)
print()

# Вычисление циклической свертки
N = 4
a = np.array([1, 2, 3, 4], dtype=int)
b = np.array([3, 2, 1, 0], dtype=int)
bi = b[::-1]
print(bi)
ab = []
for i in range(N):
    br = np.roll(bi, i+1)
    sm = np.sum(a * br)
    print(br)
    print(sm)
    ab.append(int(sm))
print(ab)

# циклическая свертка с помощью БПФ
cv = np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
print(cv)
ab_check = np.all(ab == cv)
print(ab_check)