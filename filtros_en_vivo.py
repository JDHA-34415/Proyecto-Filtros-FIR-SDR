import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import adi
import sys

# ==========================================
# 1. CONFIGURACIÓN DEL ADALM-PLUTO
# ==========================================
print("Conectando con ADALM-Pluto...")
try:
    sdr = adi.Pluto("ip:192.168.2.1")
except Exception as e:
    print(f"Error al conectar con PlutoSDR: {e}")
    sys.exit(1)

sdr.sample_rate = int(1e6)          
sdr.rx_lo = int(900e6)              
sdr.tx_lo = int(900e6)              
sdr.rx_rf_bandwidth = int(1e6)      
sdr.rx_buffer_size = 1024 * 8       

# Ganancias manuales optimizadas para poder ver el efecto de mover las antenas
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 60   
sdr.tx_hardwaregain_chan0 = 0    

# ==========================================
# 2. GENERACIÓN DE SEÑAL MULTI-TONO 
# (50 kHz, 200 kHz y 300 kHz)
# ==========================================
fs = sdr.sample_rate

# AUMENTAMOS EL BUFFER A 10,000 MUESTRAS
# Esto garantiza ciclos exactos (500, 2000 y 3000) para evitar cortes de fase
N_muestras = 10000 
t = np.arange(N_muestras) / fs

tono1 = np.exp(2j * np.pi * 50000 * t)   
tono2 = np.exp(2j * np.pi * 200000 * t)  
tono3 = np.exp(2j * np.pi * 300000 * t)  

tx_signal = tono1 + tono2 + tono3
# Escalamiento para la transmisión del PlutoSDR
tx_signal = tx_signal * (2**14 / np.max(np.abs(tx_signal)))

# Convertimos a complejo de 64 bits para evitar errores de tipo en la librería
tx_signal = tx_signal.astype(np.complex64)

sdr.tx_cyclic = True
sdr.tx(tx_signal)

# ==========================================
# 3. DISEÑO DE FILTROS FIR (65 taps, Hamming)
# ==========================================
numtaps = 65  # Impar para permitir el paso alto

coef_paso_bajo = signal.firwin(numtaps, 100000, pass_zero='lowpass', fs=fs)
coef_paso_alto = signal.firwin(numtaps, 200000, pass_zero='highpass', fs=fs)
coef_paso_banda = signal.firwin(numtaps, [150000, 250000], pass_zero='bandpass', fs=fs)

# Variables globales para alternar filtros
filtro_actual = coef_paso_banda 
nombre_filtro = "Paso Banda (150-250 kHz)"

# ==========================================
# 4. INTERACTIVIDAD (Cambiar filtros en vivo)
# ==========================================
def on_key(event):
    global filtro_actual, nombre_filtro
    
    # Imprime en la terminal la tecla exacta presionada para confirmar
    print(f"[*] Tecla detectada en la gráfica: '{event.key}'") 
    
    # Acepta teclado numérico y números normales
    if event.key in ['1', 'numpad1']:
        filtro_actual = coef_paso_bajo
        nombre_filtro = "Paso Bajo (< 100 kHz)"
        print(f"   ---> Aplicando: {nombre_filtro}")
        
    elif event.key in ['2', 'numpad2']:
        filtro_actual = coef_paso_alto
        nombre_filtro = "Paso Alto (> 200 kHz)"
        print(f"   ---> Aplicando: {nombre_filtro}")
        
    elif event.key in ['3', 'numpad3']:
        filtro_actual = coef_paso_banda
        nombre_filtro = "Paso Banda (150 - 250 kHz)"
        print(f"   ---> Aplicando: {nombre_filtro}")
    
    # Actualiza el título inmediatamente
    ax.set_title(f"SDR en Vivo | Filtro Activo: {nombre_filtro}", fontsize=14, fontweight='bold')
    fig.canvas.draw_idle()

# ==========================================
# 5. CONFIGURACIÓN DE GRÁFICA Y PROMEDIADO
# ==========================================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.mpl_connect('key_press_event', on_key) 

line_antes, = ax.plot([], [], label='Señal RX (Original con Ruido)', color='#1f77b4', alpha=0.6)
line_despues, = ax.plot([], [], label='Señal RX (Filtrada FIR)', color='#2ca02c', linewidth=2)

ax.set_title(f"SDR en Vivo | Filtro Activo: {nombre_filtro}", fontsize=14, fontweight='bold')
ax.set_xlabel("Frecuencia (Hz)", fontsize=12)
ax.set_ylabel("Magnitud Suavizada (dB)", fontsize=12)
ax.set_xlim(-fs/2, fs/2)
ax.set_ylim(-20, 80)
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.7)

# Variables para suavizado exponencial
alfa = 0.2
fft_orig_suavizada = None
fft_filt_suavizada = None
ventana_hanning = np.hanning(1024 * 8)

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
print("\n" + "="*40)
print("  INICIO DE TRANSMISIÓN Y RECEPCIÓN")
print("="*40)
print(" IMPORTANTE: ¡Haz CLIC en la ventana de la gráfica primero!")
print(" CONTROLES EN VIVO:")
print("   [1] -> Activar Filtro Paso Bajo (Mantiene 50 kHz)")
print("   [2] -> Activar Filtro Paso Alto (Mantiene 300 kHz)")
print("   [3] -> Activar Filtro Paso Banda (Mantiene 200 kHz)")
print(" Presiona Ctrl+C en esta terminal para salir.")
print("="*40 + "\n")

try:
    while True:
        samples = sdr.rx()
        filtered_samples = signal.lfilter(filtro_actual, 1.0, samples)
        
        # FFT con ventana de Hanning
        fft_orig_cruda = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples * ventana_hanning))) + 1e-10)
        fft_filt_cruda = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(filtered_samples * ventana_hanning))) + 1e-10)
        
        # Suavizado Exponencial
        if fft_orig_suavizada is None:
            fft_orig_suavizada = fft_orig_cruda
            fft_filt_suavizada = fft_filt_cruda
        else:
            fft_orig_suavizada = alfa * fft_orig_cruda + (1 - alfa) * fft_orig_suavizada
            fft_filt_suavizada = alfa * fft_filt_cruda + (1 - alfa) * fft_filt_suavizada
            
        freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/fs))
        
        # Actualización de gráfica
        line_antes.set_data(freqs, fft_orig_suavizada)
        line_despues.set_data(freqs, fft_filt_suavizada)
        
        plt.draw()
        plt.pause(0.01)

except KeyboardInterrupt:
    print("\nDeteniendo hardware SDR de forma segura...")
finally:
    sdr.tx_destroy()
    plt.ioff()
    plt.show()
    print("Finalizado.")
