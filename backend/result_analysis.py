import os
import glob
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
image_folder = './'  # Folder where your images are stored
output_path = './stitched_output.png'  # Path where stitched image will be saved

pattern = os.path.join(image_folder, "val_*")

# Load images
image_paths = sorted(glob.glob(pattern))
images = []

for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for plotting
        images.append(img)

# Check if images were loaded
if not images:
    raise ValueError("No images found starting with 'train_' in the specified folder.")

# --- Resize all images to the same size ---
h, w, c = images[0].shape
target_size = (w, h)

images = [cv2.resize(img, target_size) for img in images]

# --- Create square grid ---
num_images = len(images)
grid_size = math.ceil(math.sqrt(num_images))

# Pad with blank images if needed
blank_image = np.zeros((h, w, c), dtype=np.uint8)
while len(images) < grid_size * grid_size:
    images.append(blank_image)

# Stack images into grid
rows = []
for i in range(0, len(images), grid_size):
    row_images = images[i:i+grid_size]
    row = np.hstack(row_images)
    rows.append(row)

stitched_image = np.vstack(rows)

# --- Save stitched image to file ---
# Note: OpenCV saves images in BGR, so we must convert back
stitched_image_bgr = cv2.cvtColor(stitched_image, cv2.COLOR_RGB2BGR)
cv2.imwrite(output_path, stitched_image_bgr)
print(f"Stitched image saved at: {output_path}")

## --- Optional: Plot stitched image ---
#plt.figure(figsize=(12, 12))
#plt.imshow(stitched_image)
#plt.axis('off')
#plt.title('Square-Stitched Images from train_*')
#plt.show()
