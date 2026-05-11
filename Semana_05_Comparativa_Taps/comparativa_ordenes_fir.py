import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import adi  # Librería para la comunicación con el hardware ADALM-Pluto

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS GENERALES
# ==========================================
fs = 1e6           # Frecuencia de muestreo: 1 MHz
cutoff = 100e3     # Frecuencia de corte deseada: 100 kHz
taps_list = [32, 64, 128]  # Lista de órdenes (número de coeficientes) a comparar
ip_pluto = "ip:192.168.2.1" # Dirección IP del dispositivo SDR

# =====================================================
# 2. DISEÑO Y COMPARATIVA DE LA RESPUESTA EN FRECUENCIA
# =====================================================
plt.figure(figsize=(12, 6))

for n_taps in taps_list:
    # Generación de coeficientes del filtro FIR usando el método de enventanado
    # Se utiliza la ventana de Hamming como se especifica en el diseño
    coeffs = signal.firwin(n_taps, cutoff, fs=fs, window='hamming')
    
    # Cálculo de la respuesta en frecuencia del filtro (magnitud y fase)
    # worN=8000 define la resolución de la gráfica
    w, h = signal.freqz(coeffs, worN=8000)
    
    # Conversión de radianes a frecuencia real (Hz) y de magnitud lineal a Decibelios (dB)
    freq = (w * fs) / (2 * np.pi)
    mag_db = 20 * np.log10(np.abs(h))
    
    # Graficar la respuesta de magnitud para cada orden
    plt.plot(freq, mag_db, label=f'Orden: {n_taps} taps')

plt.title('Respuesta en Frecuencia del Filtro FIR (Comparativa de Taps)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.axvline(cutoff, color='red', linestyle='--', label='Frecuencia de Corte (100kHz)')
plt.ylim(-80, 5) # Límite inferior en -80dB para observar la atenuación
plt.grid(True)
plt.legend()

# ==========================================
# 3. CONEXIÓN Y CAPTURA CON EL ADALM-PLUTO
# ==========================================
try:
    # Inicialización del hardware
    sdr = adi.Pluto(ip_pluto)
    sdr.sample_rate = int(fs)
    sdr.rx_lo = int(100e6)      # Frecuencia central de recepción (ej. 100 MHz)
    sdr.rx_buffer_size = 1024   # Tamaño del bloque de datos capturados
    
    print("Capturando datos reales desde el ADALM-Pluto...")
    rx_signal = sdr.rx()        # Recepción de muestras complejas (I/Q)
    sdr.destroy()               # Liberar el hardware después de la captura
except Exception as e:
    print(f"Hardware no detectado ({e}). Generando señal de prueba sintética...")
    # Generación de una señal con tres tonos (50k, 200k, 300k Hz) para pruebas
    t = np.arange(1024) / fs
    rx_signal = (np.sin(2 * np.pi * 50e3 * t) + 
                 np.sin(2 * np.pi * 200e3 * t) + 
                 np.sin(2 * np.pi * 300e3 * t))

# =====================================================
# 4. PROCESAMIENTO (FFT) Y VISUALIZACIÓN DE RESULTADOS
# =====================================================
plt.figure(figsize=(12, 8))

# Cálculo de la FFT de la señal original para ver su espectro antes del filtro
f_axis = np.fft.fftfreq(len(rx_signal), 1/fs)
fft_orig = 20 * np.log10(np.abs(np.fft.fft(rx_signal))/len(rx_signal))

# Gráfica superior: Comparación general de espectros
plt.subplot(2, 1, 1)
plt.plot(np.fft.fftshift(f_axis), np.fft.fftshift(fft_orig), color='gray', alpha=0.4, label='Señal Original')

colors = ['blue', 'green', 'orange']
for i, n_taps in enumerate(taps_list):
    # Re-diseño de los coeficientes para el filtrado
    coeffs = signal.firwin(n_taps, cutoff, fs=fs, window='hamming')
    
    # Aplicación del filtro a la señal capturada (Convolución en el tiempo)
    filtered_sig = signal.lfilter(coeffs, 1.0, rx_signal)
    
    # Cálculo del espectro de la señal ya filtrada
    fft_filt = 20 * np.log10(np.abs(np.fft.fft(filtered_sig))/len(filtered_sig))
    
    plt.plot(np.fft.fftshift(f_axis), np.fft.fftshift(fft_filt), color=colors[i], label=f'Filtrada {n_taps} taps')

plt.title('Espectro de Frecuencia: Antes vs Después del Filtrado')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()

# Gráfica inferior: Zoom en la zona de transición para notar la diferencia entre taps
plt.subplot(2, 1, 2)
for i, n_taps in enumerate(taps_list):
    coeffs = signal.firwin(n_taps, cutoff, fs=fs, window='hamming')
    filtered_sig = signal.lfilter(coeffs, 1.0, rx_signal)
    fft_filt = 20 * np.log10(np.abs(np.fft.fft(filtered_sig))/len(filtered_sig))
    plt.plot(np.fft.fftshift(f_axis), np.fft.fftshift(fft_filt), color=colors[i], label=f'{n_taps} taps')

plt.xlim(0, 400e3) # Enfocar el análisis entre 0 y 400 kHz
plt.title('Zoom en la Banda de Transición')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.tight_layout()
plt.show()
