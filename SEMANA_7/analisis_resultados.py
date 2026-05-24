import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import time

# ==========================================
# 1. CONFIGURACIÓN DE PARÁMETROS UNIFICADOS
# ==========================================
fs = 1e6            # Frecuencia de muestreo: 1 MHz
num_taps = 64       # Orden base requerido
ventana = 'hamming' # Ventana especificada

f_bajo = 50e3       # Tono 1: 50 kHz
f_banda = 200e3     # Tono 2: 200 kHz
f_alto = 300e3      # Tono 3: 300 kHz

# Sincronizado a N=2000 muestras para mantener coherencia exacta con la Semana 6
N = 2000            
t = np.arange(N) / fs
rx_signal = (np.sin(2 * np.pi * f_bajo * t) + 
             np.sin(2 * np.pi * f_banda * t) + 
             np.sin(2 * np.pi * f_alto * t))

# ==========================================
# 2. DISEÑO DE FILTROS OFICIALES
# ==========================================
b_lpf = signal.firwin(num_taps, 100e3, fs=fs, window=ventana)
b_hpf = signal.firwin(num_taps + 1, 200e3, pass_zero='highpass', fs=fs, window=ventana)
b_bpf = signal.firwin(num_taps + 1, [150e3, 250e3], pass_zero='bandpass', fs=fs, window=ventana)

filtros = {
    'Paso Bajo (LPF)': b_lpf,
    'Paso Alto (HPF)': b_hpf,
    'Paso Banda (BPF)': b_bpf
}

# Diccionarios para almacenar las métricas recolectadas
tiempos_microsegundos = {}
atenuaciones_db = {}
rizados_db = {}

# ==========================================
# 3. ALGORITMO DE AUDITORÍA Y BENCHMARKING
# ==========================================
def calcular_magnitud_bin(senal, frec_objetivo, fs_rate):
    fft_vals = np.abs(np.fft.fft(senal)) / len(senal)
    frecuencias = np.fft.fftfreq(len(senal), d=1/fs_rate)
    idx = np.argmin(np.abs(frecuencias[:len(senal)//2] - frec_objetivo))
    return 20 * np.log10(fft_vals[idx] + 1e-12)

# Magnitudes de la señal original (Puntos de referencia 0 dB)
mag_orig_50 = calcular_magnitud_bin(rx_signal, f_bajo, fs)
mag_orig_200 = calcular_magnitud_bin(rx_signal, f_banda, fs)
mag_orig_300 = calcular_magnitud_bin(rx_signal, f_alto, fs)

print("Ejecutando pruebas de estrés computacional...")

for nombre, coeffs in filtros.items():
    # MEDICIÓN DE TIEMPO: Se ejecuta el filtro 500 veces para obtener un promedio estadístico estable
    t_start = time.perf_counter()
    for _ in range(500):
        filtered_sig = signal.lfilter(coeffs, 1.0, rx_signal)
    t_end = time.perf_counter()
    
    # Calcular el tiempo promedio de una sola operación en microsegundos (µs)
    tiempos_microsegundos[nombre] = ((t_end - t_start) / 500) * 1e6
    
    # Extracción de Métricas de Atenuación y Rizado (Coherentes con Semana 6)
    if nombre == 'Paso Bajo (LPF)':
        rizados_db[nombre] = calcular_magnitud_bin(filtered_sig, f_bajo, fs) - mag_orig_50
        atenuaciones_db[nombre] = calcular_magnitud_bin(filtered_sig, f_alto, fs) - mag_orig_300
    elif nombre == 'Paso Alto (HPF)':
        rizados_db[nombre] = calcular_magnitud_bin(filtered_sig, f_alto, fs) - mag_orig_300
        atenuaciones_db[nombre] = calcular_magnitud_bin(filtered_sig, f_bajo, fs) - mag_orig_50
    elif nombre == 'Paso Banda (BPF)':
        rizados_db[nombre] = calcular_magnitud_bin(filtered_sig, f_banda, fs) - mag_orig_200
        at_inf = calcular_magnitud_bin(filtered_sig, f_bajo, fs) - mag_orig_50
        at_sup = calcular_magnitud_bin(filtered_sig, f_alto, fs) - mag_orig_300
        # Se reporta la peor atenuación de las dos bandas de rechazo para ser rigurosos
        atenuaciones_db[nombre] = max(at_inf, at_sup)

# ==========================================
# 4. DESPLIEGUE DE TABLA COMPARATIVA EN CONSOLA
# ==========================================
print("\n" + "="*85)
print("     TABLA COMPARATIVA DE RENDIMIENTO TECNOLÓGICO - SEMANA 7 (FASE D)")
print("="*85)
print(f"{'Arquitectura de Filtro':<25} | {'Ripple (Passband)':<18} | {'Atenuación (Stopband)':<22} | {'Tiempo de Ejecución':<20}")
print("-"*85)
for nombre in filtros.keys():
    print(f"{nombre:<25} | {rizados_db[nombre]:>14.2f} dB | {atenuaciones_db[nombre]:>18.2f} dB | {tiempos_microsegundos[nombre]:>16.2f} µs")
print("="*85)

# ==========================================
# 5. GENERACIÓN DEL DASHBOARD DE BARRAS COMPARATIVAS
# ==========================================
nombres = list(filtros.keys())
tiempos = list(tiempos_microsegundos.values())
atenuaciones_absolutas = [abs(val) for val in atenuaciones_db.values()]

plt.figure(figsize=(14, 6))

# --- GRÁFICA 1: TIEMPO DE PROCESAMIENTO (EFICIENCIA COMPUTACIONAL) ---
plt.subplot(1, 2, 1)
barras_t = plt.bar(nombres, tiempos, color=['#1f77b4', '#2ca02c', '#ff7f0e'], width=0.4, edgecolor='black')
plt.title('Eficiencia Computacional: Tiempo de Procesamiento', fontsize=11, fontweight='bold')
plt.ylabel('Tiempo de Ejecución Promedio [µs]')
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Etiquetas de valor en las barras de tiempo
for bar in barras_t:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + (max(tiempos)*0.02), f'{yval:.2f} µs', ha='center', va='bottom', fontweight='bold')

# --- GRÁFICA 2: ATENUACIÓN EFECTIVA VS OBJETIVO DEL ALCANCE ---
plt.subplot(1, 2, 2)
barras_a = plt.bar(nombres, atenuaciones_absolutas, color=['#aec7e8', '#98df8a', '#ffbb78'], width=0.4, edgecolor='black')
plt.axhline(40, color='red', linestyle='--', linewidth=1.5, label='Límite Mínimo Requerido (> 40 dB)')
plt.title('Respuesta en Frecuencia: Atenuación Efectiva en Stopband', fontsize=11, fontweight='bold')
plt.ylabel('Atenuación Absoluta Mínima [dB]')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(loc='lower right')

# Etiquetas de valor en las barras de atenuación
for bar in barras_a:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f} dB', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Guardar automáticamente el entregable de la Semana 7
plt.savefig('Semana_07_Analisis_Comparativo.png', dpi=300)
print("\n¡Gráfica comparativa 'Semana_07_Analisis_Comparativo.png' exportada con éxito!")
plt.show()
