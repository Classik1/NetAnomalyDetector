import numpy as np
import matplotlib.pyplot as plt

A = [1, 0.5, 0.25]   
F = [0.1, 0.1, 0.4]   
fs = 100                
T = 10                 
N = int(fs * T)
t = np.linspace(0, T, N)

def generate_signal(t, A, F):
    signal = np.zeros_like(t)
    for a, f in zip(A, F):
        signal += a * np.sin(2 * np.pi * f * t)
    return signal

raw_signal = generate_signal(t, A, F)

np.random.seed(42)
noise = 0.2 * np.random.randn(len(t))
noisy_signal = raw_signal + noise

buffer_size = 50
buffer = []

def ema_filter(data, alpha=0.3):
    result = []
    prev = data[0]
    for value in data:
        ema = alpha * value + (1 - alpha) * prev
        result.append(ema)
        prev = ema
    return result

alpha = 0.2 
filtered_signal = ema_filter(noisy_signal, alpha)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t, raw_signal, label='Исходный сигнал (чистый)', linewidth=2)
plt.plot(t, noisy_signal, label='Сигнал + шум', alpha=0.7)
plt.legend()
plt.title('До фильтрации')
plt.xlabel('Время (с)')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, noisy_signal, label='Зашумленный сигнал', alpha=0.5)
plt.plot(t, filtered_signal, label='После EMA фильтра', linewidth=2, color='red')
plt.legend()
plt.title('После фильтрации (скользящее среднее)')
plt.xlabel('Время (с)')
plt.grid(True)

plt.tight_layout()
plt.savefig('lab2_result.png')
plt.show()

print("Сигнал сгенерирован, отфильтрован и сохранен в lab2_result.png")