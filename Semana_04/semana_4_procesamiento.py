import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import adi

# 1. Parámetros de Diseño (Según Especificaciones del Proyecto)
fs = 1e6  # Frecuencia de muestreo (1 MHz para cubrir hasta 300 kHz cómodamente)
num_taps = 65
ventana = 'hamming'

# 2. Configuración del ADALM-Pluto
try:
    sdr = adi.Pluto("ip:192.168.2.1")
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(1e9) # 1 GHz frecuencia central
    sdr.rx_buffer_size = 1024 * 8
except:
    print("Error: No se detectó el Pluto. Asegúrate de estar conectado.")

# 3. Definición de Filtros FIR
# Filtro Paso Bajo: Corte 100 kHz
h_lp = signal.firwin(num_taps, 100e3, fs=fs, window=ventana)

# Filtro Paso Alto: Corte 200 kHz
h_hp = signal.firwin(num_taps, 200e3, fs=fs, window=ventana, pass_zero=False)

# Filtro Paso Banda: 150-250 kHz
h_bp = signal.firwin(num_taps, [150e3, 250e3], fs=fs, window=ventana, pass_zero=False)

# 4. Captura de Señales I/Q
print("Capturando señal multi-tono...")
samples = sdr.rx()

# 5. Aplicación de los Filtros
filt_lp = signal.lfilter(h_lp, 1.0, samples)
filt_hp = signal.lfilter(h_hp, 1.0, samples)
filt_bp = signal.lfilter(h_bp, 1.0, samples)

# 6. Función para Graficar Espectros (FFT)
def plot_spectrum(sig, title, pos):
    n = len(sig)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1/fs))
    # Normalización y escala logarítmica para verificar atenuación > 40 dB
    espectro = np.fft.fftshift(np.abs(np.fft.fft(sig)))
    espectro_db = 20 * np.log10(espectro / np.max(espectro)) 
    
    plt.subplot(2, 2, pos)
    plt.plot(freqs/1e3, espectro_db)
    plt.title(title)
    plt.ylabel('Magnitud (dB)')
    plt.xlabel('Frecuencia (kHz)')
    plt.ylim([-80, 5]) # Rango para observar el Stopband
    plt.grid(True)

# 7. Visualización de Resultados
plt.figure(figsize=(12, 10))
plot_spectrum(samples, "Original: 50+200+300 kHz", 1)
plot_spectrum(filt_lp, "Filtrada: Paso Bajo (Solo 50 kHz)", 2)
plot_spectrum(filt_hp, "Filtrada: Paso Alto (Solo 300 kHz)", 3)
plot_spectrum(filt_bp, "Filtrada: Paso Banda (Solo 200 kHz)", 4)

plt.tight_layout()
plt.show()