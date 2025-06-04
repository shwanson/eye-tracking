import os
import re

# Directory containing the CSV files
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Regex patterns
proband_pattern = re.compile(r'Proband(\d+)')
id_pattern = re.compile(r'id(\d+)')

for filename in os.listdir(DATA_DIR):
    if filename.endswith('.csv'):
        new_filename = filename
        # Replace ProbandX with PXXX
        def repl_proband(match):
            return f"P{int(match.group(1)):03d}"
        new_filename = proband_pattern.sub(repl_proband, new_filename)
        # Replace idY with idYYY
        def repl_id(match):
            return f"id{int(match.group(1)):03d}"
        new_filename = id_pattern.sub(repl_id, new_filename)
        # Rename if changed
        if new_filename != filename:
            src = os.path.join(DATA_DIR, filename)
            dst = os.path.join(DATA_DIR, new_filename)
            print(f"Renaming: {filename} -> {new_filename}")
            os.rename(src, dst) 