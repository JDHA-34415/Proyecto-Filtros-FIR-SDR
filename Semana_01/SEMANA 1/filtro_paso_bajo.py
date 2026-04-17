
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs = 1e6            
cutoff = 100e3      
numtaps = 32      
window = 'hamming'  

taps = signal.firwin(numtaps, cutoff, fs=fs, window=window)

w, h = signal.freqz(taps, worN=8000)
f = w * fs / (2 * np.pi) 


plt.figure(figsize=(10, 6))


plt.plot(f, 20 * np.log10(abs(h)), 'b')
plt.title('Respuesta en Frecuencia - Filtro Paso Bajo (32 taps)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Amplitud [dB]')
plt.grid(which='both', linestyle='-', alpha=0.5)

plt.axvline(cutoff, color='red', linestyle='--', label='Corte (100 kHz)')
plt.axhline(-3, color='green', linestyle=':', label='Punto -3dB')
plt.ylim([-80, 5]) 
plt.legend()

plt.show()

print(f"Coeficientes del filtro ({numtaps} taps):")
print(taps)
