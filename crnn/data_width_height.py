from PIL import Image
import os

# List of PNG images (update the path accordingly)
image_folder_path = 'A:\\aaa\\data\\test'
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.png')]

max_width = 0
max_height = 0
max_width_image = ""
max_height_image = ""

# Iterate through each image to find the maximum width and height
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    with Image.open(image_path) as img:
        width, height = img.size
        # Check for maximum width
        if width > max_width:
            max_width = width
            max_width_image = image_file
        # Check for maximum height
        if height > max_height:
            max_height = height
            max_height_image = image_file

print(f"Maximum Width: {max_width} (Image: {max_width_image})")
print(f"Maximum Height: {max_height} (Image: {max_height_image})")

