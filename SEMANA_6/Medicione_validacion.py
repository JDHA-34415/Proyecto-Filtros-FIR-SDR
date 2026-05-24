import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS GENERALES
# ==========================================
fs = 1e6            # Frecuencia de muestreo: 1 MHz
num_taps = 64       # Orden base requerido (64 taps)
ventana = 'hamming' # Ventana especificada en los lineamientos

# Frecuencias de los tonos de prueba obligatorios según el alcance
f_bajo = 50e3       # Tono 1: 50 kHz
f_banda = 200e3     # Tono 2: 200 kHz
f_alto = 300e3      # Tono 3: 300 kHz

# -----------------------------------------------------------------
# GENERACIÓN DE SEÑAL DE VALIDACIÓN MATEMÁTICA PERFECTA
# -----------------------------------------------------------------
N = 2000            
t = np.arange(N) / fs
rx_signal = (np.sin(2 * np.pi * f_bajo * t) + 
             np.sin(2 * np.pi * f_banda * t) + 
             np.sin(2 * np.pi * f_alto * t))

# ==========================================
# 2. DISEÑO DE FILTROS CON RIGOR DSP
# ==========================================
b_lpf = signal.firwin(num_taps, 100e3, fs=fs, window=ventana)

num_taps_impar = num_taps + 1
b_hpf = signal.firwin(num_taps_impar, 200e3, pass_zero='highpass', fs=fs, window=ventana)
b_bpf = signal.firwin(num_taps_impar, [150e3, 250e3], pass_zero='bandpass', fs=fs, window=ventana)

senal_lpf = signal.lfilter(b_lpf, 1.0, rx_signal)
senal_hpf = signal.lfilter(b_hpf, 1.0, rx_signal)
senal_bpf = signal.lfilter(b_bpf, 1.0, rx_signal)

# ==========================================
# 3. ALGORITMO DE MEDICIÓN ESPECTRAL (FFT)
# ==========================================
def calcular_magnitud_bin(senal, frec_objetivo, fs_rate):
    fft_vals = np.abs(np.fft.fft(senal)) / len(senal)
    frecuencias = np.fft.fftfreq(len(senal), d=1/fs_rate)
    idx = np.argmin(np.abs(frecuencias[:len(senal)//2] - frec_objetivo))
    return 20 * np.log10(fft_vals[idx] + 1e-12)

mag_orig_50 = calcular_magnitud_bin(rx_signal, f_bajo, fs)
mag_orig_200 = calcular_magnitud_bin(rx_signal, f_banda, fs)
mag_orig_300 = calcular_magnitud_bin(rx_signal, f_alto, fs)

# ==========================================
# 4. CONTROL DE CALIDAD INTERNO (AUDITORÍA)
# ==========================================
print("\n" + "="*75)
print("       SISTEMA AUTOMÁTICO DE AUDITORÍA DE MÉTRICAS - SEMANA 6")
print("="*75)

ganancia_lpf_50 = calcular_magnitud_bin(senal_lpf, f_bajo, fs) - mag_orig_50
atenuacion_lpf_300 = calcular_magnitud_bin(senal_lpf, f_alto, fs) - mag_orig_300
status_lpf = "PASÓ (CUMPLE)" if (abs(ganancia_lpf_50) < 1.0 and atenuacion_lpf_300 <= -40.0) else "FALLÓ"

print(f"[FILTRO PASO BAJO] (Especificación: 64 Taps - Corte: 100 kHz)")
print(f"  -> Rizado en Banda de Paso (50 kHz):      {ganancia_lpf_50:.2f} dB  [Objetivo: < 1 dB]")
print(f"  -> Atenuación en Banda de Rechazo (300 kHz): {atenuacion_lpf_300:.2f} dB [Objetivo: > 40 dB]")
print(f"  -> CONTROL DE CALIDAD INTERNO: {status_lpf}")
print("-"*75)

atenuacion_hpf_50 = calcular_magnitud_bin(senal_hpf, f_bajo, fs) - mag_orig_50
ganancia_hpf_300 = calcular_magnitud_bin(senal_hpf, f_alto, fs) - mag_orig_300
status_hpf = "PASÓ (CUMPLE)" if (abs(ganancia_hpf_300) < 1.0 and atenuacion_hpf_50 <= -40.0) else "FALLÓ"

print(f"[FILTRO PASO ALTO] (Especificación: 65 Taps - Corte: 200 kHz)")
print(f"  -> Atenuación en Banda de Rechazo (50 kHz):  {atenuacion_hpf_50:.2f} dB [Objetivo: > 40 dB]")
print(f"  -> Rizado en Banda de Paso (300 kHz):     {ganancia_hpf_300:.2f} dB  [Objetivo: < 1 dB]")
print(f"  -> CONTROL DE CALIDAD INTERNO: {status_hpf}")
print("-"*75)

atenuacion_bpf_50 = calcular_magnitud_bin(senal_bpf, f_bajo, fs) - mag_orig_50
ganancia_bpf_200 = calcular_magnitud_bin(senal_bpf, f_banda, fs) - mag_orig_200
atenuacion_bpf_300 = calcular_magnitud_bin(senal_bpf, f_alto, fs) - mag_orig_300
cumple_bpf = (abs(ganancia_bpf_200) < 1.0 and atenuacion_bpf_50 <= -40.0 and atenuacion_bpf_300 <= -40.0)
status_bpf = "PASÓ (CUMPLE)" if cumple_bpf else "FALLÓ"

print(f"[FILTRO PASO BANDA] (Especificación: 65 Taps - Banda: 150-250 kHz)")
print(f"  -> Atenuación en Rechazo Inferior (50 kHz): {atenuacion_bpf_50:.2f} dB [Objetivo: > 40 dB]")
print(f"  -> Rizado en Banda de Paso (200 kHz):       {ganancia_bpf_200:.2f} dB  [Objetivo: < 1 dB]")
print(f"  -> Atenuación en Rechazo Superior (300 kHz): {atenuacion_bpf_300:.2f} dB [Objetivo: > 40 dB]")
print(f"  -> CONTROL DE CALIDAD INTERNO: {status_bpf}")
print("="*75)

# ==========================================
# 5. GENERACIÓN VISUAL DE ENTREGABLES
# ==========================================
frecuencias_khz = np.fft.fftfreq(N, d=1/fs)[:N//2] / 1e3
plt.figure(figsize=(12, 10))
fft_orig = 20 * np.log10(np.abs(np.fft.fft(rx_signal))[:N//2] / N + 1e-12)

plt.subplot(3, 1, 1)
plt.plot(frecuencias_khz, fft_orig, color='gray', alpha=0.5, label='Señal Original sin Filtrar')
plt.plot(frecuencias_khz, 20 * np.log10(np.abs(np.fft.fft(senal_lpf))[:N//2] / N + 1e-12), color='blue', label='Salida Filtro LPF')
plt.axvline(100, color='red', linestyle='--', label='Corte LPF (100 kHz)')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(frecuencias_khz, fft_orig, color='gray', alpha=0.5, label='Señal Original sin Filtrar')
plt.plot(frecuencias_khz, 20 * np.log10(np.abs(np.fft.fft(senal_hpf))[:N//2] / N + 1e-12), color='green', label='Salida Filtro HPF')
plt.axvline(200, color='red', linestyle='--', label='Corte HPF (200 kHz)')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(frecuencias_khz, fft_orig, color='gray', alpha=0.5, label='Señal Original sin Filtrar')
plt.plot(frecuencias_khz, 20 * np.log10(np.abs(np.fft.fft(senal_bpf))[:N//2] / N + 1e-12), color='orange', label='Salida Filtro BPF')
plt.axvline(150, color='red', linestyle=':', label='Banda de Paso (150-250 kHz)')
plt.axvline(250, color='red', linestyle=':')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('Semana_06_Validacion_Espectral_Correcto.png', dpi=300)
plt.show()
