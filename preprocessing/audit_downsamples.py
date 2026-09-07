import os
import openslide
from collections import Counter

SLIDES_DIR = "/cs/student/project_msc/2025/aibh/iconstan/slides"

svs_files = []
for root, dirs, files in os.walk(SLIDES_DIR):
    for f in files:
        if f.endswith(".svs"):
            svs_files.append(os.path.join(root, f))

print(f"Auditing {len(svs_files)} slides...\n")

downsample_counter = Counter()
problem_slides = []

for i, path in enumerate(svs_files):
    try:
        slide = openslide.OpenSlide(path)
        if len(slide.level_downsamples) > 1:
            actual_ds1 = slide.level_downsamples[1]
            rounded = round(actual_ds1, 1)
            downsample_counter[rounded] += 1
            # Flag if NOT close to 2.0 (i.e. more than 5% off)
            if abs(actual_ds1 - 2.0) > 0.1:
                problem_slides.append((os.path.basename(path), actual_ds1))
        slide.close()
    except Exception as e:
        print(f"  ERROR on {os.path.basename(path)}: {e}")

    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1}/{len(svs_files)}...")

print(f"\n=== DOWNSAMPLE FACTOR DISTRIBUTION (level 1 vs level 0) ===")
for ds, count in sorted(downsample_counter.items()):
    print(f"  {ds}x: {count} slides")

print(f"\n=== SLIDES WHERE LEVEL 1 IS NOT ~2X (i.e. affected by the bug) ===")
print(f"Total affected: {len(problem_slides)} / {len(svs_files)}")
for name, ds in problem_slides[:20]:
    print(f"  {name}: {ds:.4f}x")
if len(problem_slides) > 20:
    print(f"  ... and {len(problem_slides)-20} more")
