
import cv2
import numpy as np

# Create a simulated thermal-style background
image = np.full((400, 600), 35, dtype=np.uint8)

# Add cooler background objects
cv2.rectangle(image, (50, 80), (140, 300), 65, -1)

# Simulate a warm human-shaped object
cv2.circle(image, (300, 100), 25, 230, -1)
cv2.ellipse(image, (300, 200), (40, 75), 0, 0, 360, 210, -1)

# Simulate warm legs
cv2.rectangle(image, (275, 260), (295, 350), 190, -1)
cv2.rectangle(image, (305, 260), (325, 350), 190, -1)

# Save as a grayscale image
cv2.imwrite("real_thermal.png", image)

print("Sample thermal-style image created!")
