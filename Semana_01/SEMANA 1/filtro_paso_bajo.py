

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. Parámetros según el documento 
fs = 1e6            # Frecuencia de muestreo (1 MHz para cubrir señales de hasta 300kHz)
cutoff = 100e3      # Frecuencia de corte: 100 kHz 
numtaps = 32       # Orden: 32 taps 
window = 'hamming'  # Ventana: Hamming 

# 2. Diseño del filtro FIR usando scipy.signal.firwin 
# Nyquist es la mitad de la frecuencia de muestreo
taps = signal.firwin(numtaps, cutoff, fs=fs, window=window)

# 3. Calcular la respuesta en frecuencia 
w, h = signal.freqz(taps, worN=8000)
f = w * fs / (2 * np.pi) # Convertir radianes a Hz

# 4. Visualización 
plt.figure(figsize=(10, 6))

# Magnitud en dB
plt.plot(f, 20 * np.log10(abs(h)), 'b')
plt.title('Respuesta en Frecuencia - Filtro Paso Bajo (32 taps)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', linestyle='-', alpha=0.5)

# Verificación de métricas del documento 
plt.axvline(cutoff, color='red', linestyle='--', label='Corte (100 kHz)')
plt.axhline(-3, color='green', linestyle=':', label='Punto -3dB')
plt.ylim([-80, 5]) # Para ver la atenuación de >40dB en stopband
plt.legend()

plt.show()

# Imprimir los coeficientes resultantes (opcional para el informe) 
print(f"Coeficientes del filtro ({numtaps} taps):")
print(taps)
