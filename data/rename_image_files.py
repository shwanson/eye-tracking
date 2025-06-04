import os
import re

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'current_images')

id_pattern = re.compile(r'id(\d+)_')

for filename in os.listdir(IMG_DIR):
    if filename.endswith('.jpg'):
        new_filename = filename
        def repl_id(match):
            return f"id{int(match.group(1)):03d}_"
        new_filename = id_pattern.sub(repl_id, new_filename)
        if new_filename != filename:
            src = os.path.join(IMG_DIR, filename)
            dst = os.path.join(IMG_DIR, new_filename)
            print(f"Renaming: {filename} -> {new_filename}")
            os.rename(src, dst) 