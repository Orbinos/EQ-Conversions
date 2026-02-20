import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import freqz
import re

# =====================================================================
# 1. DSP: Biquad Filter Coefficients
# =====================================================================
def get_biquad_coefs(filter_type, fc, gain, q, fs=48000):
    """Generates RBJ biquad coefficients for standard EQ filters."""
    # Prevent nyquist crashes
    fc = np.clip(fc, 10, fs / 2 * 0.99)
    q = max(q, 0.01)

    A = 10 ** (gain / 40.0)
    w0 = 2 * np.pi * fc / fs
    alpha = np.sin(w0) / (2 * q)

    if filter_type == 'PK':
        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A
    elif filter_type == 'LSC':
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == 'HSC':
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    elif filter_type == 'HPF':
        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = (1 + np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    elif filter_type == 'LPF':
        b0 = (1 - np.cos(w0)) / 2
        b1 = 1 - np.cos(w0)
        b2 = (1 - np.cos(w0)) / 2
        a0 = 1 + alpha
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha
    else:
        return np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])

    # Normalize by a0
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def get_frequency_response(preamp, filters, freqs, fs=48000):
    """Calculates the combined magnitude response in dB."""
    total_response = np.ones_like(freqs, dtype=complex) * (10 ** (preamp / 20.0))
    for f in filters:
        b, a = get_biquad_coefs(f['type'], f['fc'], f['gain'], f['q'], fs)
        w, h = freqz(b, a, worN=freqs, fs=fs)
        total_response *= h
    return 20 * np.log10(np.abs(total_response) + 1e-10)

# =====================================================================
# 2. Parsing Input
# =====================================================================
def parse_eq_text(text):
    preamp = 0.0
    filters = []
    lines = text.strip().split('\n')
    for line in lines:
        if line.startswith('Preamp:'):
            match = re.search(r'Preamp:\s*([-\d.]+)\s*dB', line)
            if match: preamp = float(match.group(1))
        elif line.startswith('Filter'):
            match = re.search(r'ON\s+([A-Z]+)\s+Fc\s+([-\d.]+)\s+Hz\s+Gain\s+([-\d.]+)\s+dB\s+Q\s+([-\d.]+)', line)
            if match:
                filters.append({
                    'type': match.group(1),
                    'fc': float(match.group(2)),
                    'gain': float(match.group(3)),
                    'q': float(match.group(4))
                })
    return preamp, filters

# =====================================================================
# 3. Optimization Setup
# =====================================================================
# Target mappings: 0: PK, 1: LSC, 2: HPF, 3: LPF  (NO HSC allowed)
TARGET_TYPES = {0: 'PK', 1: 'LSC', 2: 'HPF', 3: 'LPF'}

def objective_function(x, target_freqs, target_magnitude, num_bands):
    """Objective: Minimize Mean Squared Error (MSE) between source and target curves."""
    preamp = x[0]
    candidate_filters = []

    for i in range(num_bands):
        idx = 1 + i * 4
        f_type_idx = int(np.round(x[idx]))
        candidate_filters.append({
            'type': TARGET_TYPES[f_type_idx],
            'fc': x[idx+1],
            'gain': x[idx+2],
            'q': x[idx+3]
        })

    candidate_magnitude = get_frequency_response(preamp, candidate_filters, target_freqs)

    # Compute Mean Squared Error. Heavily penalize deviation.
    mse = np.mean((target_magnitude - candidate_magnitude)**2)
    return mse

# =====================================================================
# 4. Main Execution
# =====================================================================
input_eq = """Preamp: -7.1 dB
Filter 1: ON PK Fc 66 Hz Gain 0.83 dB Q 1.022
Filter 2: ON PK Fc 172 Hz Gain -1.67 dB Q 0.811
Filter 3: ON LSC Fc 553 Hz Gain -3.03 dB Q 0.985
Filter 4: ON PK Fc 1516 Hz Gain -3.91 dB Q 0.627
Filter 5: ON PK Fc 3314 Hz Gain 2.74 dB Q 1.748
Filter 6: ON PK Fc 5083 Hz Gain -5.26 dB Q 1.905
Filter 7: ON HSC Fc 5276 Hz Gain 6.91 dB Q 1.020
Filter 8: ON PK Fc 8036 Hz Gain -10.80 dB Q 2.806
Filter 9: ON PK Fc 10461 Hz Gain 10.77 dB Q 1.954
Filter 10: ON PK Fc 12048 Hz Gain -12.89 dB Q 1.621"""

print("1. Parsing source 10-band EQ...")
source_preamp, source_filters = parse_eq_text(input_eq)

print("2. Generating target magnitude response curve...")
# Generate logarithmic frequencies from 20Hz to 20kHz to calculate MSE accurately
freqs = np.logspace(np.log10(20), np.log10(20000), 500)
source_mag = get_frequency_response(source_preamp, source_filters, freqs)

print("3. Setting up boundaries and constraints...")
num_target_bands = 8
bounds = []
# Preamp constraint: [-16, 6]
bounds.append((-16.0, 6.0))

# Integrality array for Differential Evolution (1 = integer, 0 = float)
# Allows the optimizer to discrete-hop between filter types
integrality = [0]

for _ in range(num_target_bands):
    bounds.append((0, 3))            # Type (Discrete 0-3)
    bounds.append((20.0, 20000.0))   # Fc (Frequency)
    bounds.append((-10.0, 10.0))     # Gain constraint: [-10, 10]
    bounds.append((0.1, 10.0))       # Q factor bounds

    integrality.extend([1, 0, 0, 0])

print("4. Running Differential Evolution Optimizer... (This may take a few minutes...)")
# popsize=20 and maxiter=2000 makes this a very heavy, thorough approximation search
result = differential_evolution(
    objective_function,
    bounds=bounds,
    integrality=integrality,
    args=(freqs, source_mag, num_target_bands),
    strategy='best1bin',
    maxiter=2000,
    popsize=20,
    tol=1e-5,
    disp=True # Prints progress
)

print("\n--- OPTIMIZATION COMPLETE ---\n")
print(f"Preamp: {result.x[0]:.2f} dB")
for i in range(num_target_bands):
    idx = 1 + i * 4
    f_type = TARGET_TYPES[int(np.round(result.x[idx]))]
    fc = result.x[idx+1]
    gain = result.x[idx+2]
    q = result.x[idx+3]
    print(f"Filter {i+1}: ON {f_type} Fc {fc:.0f} Hz Gain {gain:.2f} dB Q {q:.3f}")
