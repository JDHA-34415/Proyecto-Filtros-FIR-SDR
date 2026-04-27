import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros Globales 
fs = 1e6            # Frecuencia de muestreo (1 MHz)
numtaps = 65       # Orden: 64 taps
window = 'hamming'  # Ventana: Hamming

# --- 1. DISEÑO FILTRO PASO ALTO ---
# Frecuencia de corte: 200 kHz 
cutoff_high = 200e3
taps_high = signal.firwin(numtaps, cutoff_high, fs=fs, window=window, pass_zero=False)

# --- 2. DISEÑO FILTRO PASO BANDA ---
# Banda de paso: 150-250 kHz 
band = [150e3, 250e3]
taps_band = signal.firwin(numtaps, band, fs=fs, window=window, pass_zero=False)

# --- 3. CÁLCULO DE RESPUESTAS ---
w_h, h_h = signal.freqz(taps_high, worN=8000)
w_b, h_b = signal.freqz(taps_band, worN=8000)

# --- 4. VISUALIZACIÓN ---
plt.figure(figsize=(12, 6))

# Gráfica Paso Alto
plt.subplot(1, 2, 1)
plt.plot(w_h * fs / (2 * np.pi), 20 * np.log10(abs(h_h)), 'r')
plt.title('Filtro Paso Alto (Corte: 200 kHz)')
plt.axvline(200e3, color='black', linestyle='--')
plt.ylabel('Amplitud [dB]')
plt.grid(True)

# Gráfica Paso Banda
plt.subplot(1, 2, 2)
plt.plot(w_b * fs / (2 * np.pi), 20 * np.log10(abs(h_b)), 'g')
plt.title('Filtro Paso Banda (150-250 kHz)')
plt.axvline(150e3, color='black', linestyle='--')
plt.axvline(250e3, color='black', linestyle='--')
plt.grid(True)

plt.tight_layout()
plt.show()
