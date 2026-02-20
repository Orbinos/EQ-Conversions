import numpy as np
import re

def parse_svg_data(raw_string):
    """Extracts X/Y coordinates and the Y-transform offset dynamically."""
    # 1. Grab the translation offset so we don't have to hardcode it
    y_offset = 0.0
    transform_match = re.search(r'translate\(([^,]+),\s*([^)]+)\)', raw_string)
    if transform_match:
        y_offset = float(transform_match.group(2))
        
    # 2. Extract just the path coordinates, ignoring all other HTML fluff
    match = re.search(r'd="([^"]+)"', raw_string, re.IGNORECASE)
    path_d = match.group(1) if match else raw_string
        
    path_d = path_d.strip().upper().replace(' ', '')
    segments = path_d.replace('M', '').split('C')
    points = []
    
    start_coords = segments[0].split(',')
    points.append((float(start_coords[0]), float(start_coords[1])))
    
    for seg in segments[1:]:
        coords = seg.split(',')
        if len(coords) >= 6:
            points.append((float(coords[4]), float(coords[5])))
            
    return np.array(points), y_offset

def convert_perfect_fr(raw_svg, output_filename="perfect_fr_graph.txt"):
    # Extract points and the Y-offset dynamic to whatever string you paste
    points_array, PATH_Y_OFFSET = parse_svg_data(raw_svg)
    
    # ==========================================
    # ⚙️ 1. EXPORT OFFSET
    # ==========================================
    # Shifts the 0 dB graph up to standard measurement volume
    TARGET_OFFSET_DB = 75.0 

    # ==========================================
    # 📐 2. THE ABSOLUTE Y-AXIS KEYS 
    # ==========================================
    DB_5_Y = 130.0495
    DB_MINUS_5_Y = 184.8243
    pixels_per_db = (DB_MINUS_5_Y - DB_5_Y) / 10.0  # Exactly 5.47748
    # ==========================================

    x_raw = points_array[:, 0]
    y_raw = points_array[:, 1] + PATH_Y_OFFSET
    
    # Sort sequentially to prevent looping artifacts
    sort_idx = np.argsort(x_raw)
    x_raw = x_raw[sort_idx]
    y_raw = y_raw[sort_idx]

    # --- THE FIX: X-AXIS ---
    # Bounding exactly to the drawn path ensures the shape stays perfect
    x_min, x_max = x_raw.min(), x_raw.max()
    freqs = 20.0 * np.power((20000.0 / 20.0), (x_raw - x_min) / (x_max - x_min))

    # --- THE Y-AXIS ---
    # Absolute mapping based on your specific grid lines
    visual_db_values = 5.0 - ((y_raw - DB_5_Y) / pixels_per_db)
    final_db_values = visual_db_values + TARGET_OFFSET_DB

    # Resample to match your target file format (500 log points)
    benchmark_freqs = np.logspace(np.log10(20), np.log10(20000), num=500)
    smoothed_db = np.interp(benchmark_freqs, freqs, final_db_values)

    # Export
    with open(output_filename, 'w') as f:
        for fq, db in zip(benchmark_freqs, smoothed_db):
            f.write(f"{fq:.2f} {db:.3f}\n")
            
    print(f"✅ Absolute calibration complete! Saved {len(benchmark_freqs)} points.")

# ==========================================
# 📥 3. PASTE YOUR SVG TAG BELOW
# ==========================================
# You can now paste the entire <path ... > tag here. 
raw_svg_input = """

E.g.:
<path stroke="rgb(0, 166, 0)" transform="translate(0,529.6883122783381)" d="M15,-394.4588108108108C15,-394.4588108108108,16.3413952353083,-394.45445857793123,16.609674282369955,-394.4588108108108C16.87795332943161,-394.4631630436904,17.951069517678206,
...
-394.50169511002656,18.219348564739857,-394.51103760536563C18.487627611801507,-394.5203801007047,19.56074380004811,-394.56128623497705,19.829022847109762,-394.5709207548796C20.097301894171412,786.0339812551875,-256.0429549549549" class=""></path>

"""

# Clean and run
cleaned_svg = raw_svg_input.replace('\n', '').replace('\r', '').strip()
convert_perfect_fr(cleaned_svg)
