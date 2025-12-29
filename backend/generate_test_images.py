"""
Generate synthetic industrial defect sample images for testing.
Creates various types of defects: scratches, stains, cracks, holes.
"""
import numpy as np
import cv2
import os
from pathlib import Path

# Create test_images directory
output_dir = Path("./test_images")
output_dir.mkdir(exist_ok=True)

def create_base_texture(size=256, texture_type="metal"):
    """Create a base industrial texture."""
    img = np.ones((size, size, 3), dtype=np.float32) * 180
    
    if texture_type == "metal":
        # Brushed metal effect
        for i in range(size):
            noise = np.random.uniform(-10, 10)
            img[i, :] = img[i, :] + noise
        # Add some grain
        noise = np.random.normal(0, 5, (size, size, 3))
        img = img + noise
    
    elif texture_type == "fabric":
        # Grid pattern
        for i in range(0, size, 4):
            img[i, :] = img[i, :] - 20
            img[:, i] = img[:, i] - 20
        noise = np.random.normal(0, 8, (size, size, 3))
        img = img + noise
    
    elif texture_type == "wood":
        # Wood grain
        for i in range(size):
            wave = int(5 * np.sin(i / 10))
            shift = (i + wave) % 20
            if shift < 3:
                img[i, :] = img[i, :] - 30
        img[:, :, 0] = img[:, :, 0] - 20  # More blue = brown tint
        noise = np.random.normal(0, 3, (size, size, 3))
        img = img + noise
    
    return np.clip(img, 0, 255).astype(np.uint8)

def add_scratch(img, severity="medium"):
    """Add a scratch defect."""
    h, w = img.shape[:2]
    img = img.copy()
    
    num_scratches = {"light": 1, "medium": 2, "heavy": 4}.get(severity, 2)
    
    for _ in range(num_scratches):
        x1 = np.random.randint(10, w-10)
        y1 = np.random.randint(10, h//2)
        length = np.random.randint(50, 150)
        angle = np.random.uniform(-30, 30)
        
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = int(y1 + length * np.sin(np.radians(angle + 90)))
        
        x2 = max(0, min(w-1, x2))
        y2 = max(0, min(h-1, y2))
        
        thickness = np.random.randint(1, 3)
        color = (60, 60, 60)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    return img

def add_stain(img, severity="medium"):
    """Add a stain/contamination defect."""
    h, w = img.shape[:2]
    img = img.copy().astype(np.float32)
    
    num_stains = {"light": 1, "medium": 2, "heavy": 3}.get(severity, 2)
    
    for _ in range(num_stains):
        cx = np.random.randint(30, w-30)
        cy = np.random.randint(30, h-30)
        radius = np.random.randint(10, 40)
        
        # Create irregular stain
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1.0, -1)
        
        # Make irregular
        kernel = np.random.randint(3, 7)
        if kernel % 2 == 0:
            kernel += 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
        
        # Darken stain area
        stain_intensity = np.random.uniform(40, 80)
        for c in range(3):
            img[:, :, c] = img[:, :, c] - mask * stain_intensity
    
    return np.clip(img, 0, 255).astype(np.uint8)

def add_crack(img, severity="medium"):
    """Add a crack defect."""
    h, w = img.shape[:2]
    img = img.copy()
    
    # Start point
    x = np.random.randint(w//4, 3*w//4)
    y = np.random.randint(10, h//3)
    
    points = [(x, y)]
    length = {"light": 50, "medium": 100, "heavy": 150}.get(severity, 100)
    
    for _ in range(length):
        dx = np.random.randint(-2, 3)
        dy = np.random.randint(1, 4)
        x = int(np.clip(x + dx, 0, w-1))
        y = int(np.clip(y + dy, 0, h-1))
        points.append((x, y))
    
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (30, 30, 30), 1)
    
    return img

def add_hole(img, severity="medium"):
    """Add a hole/pit defect."""
    h, w = img.shape[:2]
    img = img.copy()
    
    num_holes = {"light": 1, "medium": 3, "heavy": 5}.get(severity, 3)
    
    for _ in range(num_holes):
        cx = np.random.randint(20, w-20)
        cy = np.random.randint(20, h-20)
        radius = np.random.randint(3, 12)
        
        cv2.circle(img, (cx, cy), radius, (20, 20, 20), -1)
        cv2.circle(img, (cx-1, cy-1), max(1, radius-1), (40, 40, 40), 1)
    
    return img

def create_normal_image(texture_type="metal"):
    """Create a defect-free image."""
    return create_base_texture(256, texture_type)

# Generate sample images
print("Generating synthetic industrial defect images...")

# Normal images (defect-free)
for i, texture in enumerate(["metal", "fabric", "wood"]):
    img = create_normal_image(texture)
    cv2.imwrite(str(output_dir / f"normal_{texture}_{i+1}.png"), img)
    print(f"Created: normal_{texture}_{i+1}.png")

# Defective images
defect_types = [
    ("scratch", add_scratch),
    ("stain", add_stain),
    ("crack", add_crack),
    ("hole", add_hole),
]

for defect_name, defect_func in defect_types:
    for severity in ["light", "medium", "heavy"]:
        base = create_base_texture(256, "metal")
        defective = defect_func(base, severity)
        filename = f"defect_{defect_name}_{severity}.png"
        cv2.imwrite(str(output_dir / filename), defective)
        print(f"Created: {filename}")

# Mixed defects
base = create_base_texture(256, "metal")
mixed = add_scratch(base, "medium")
mixed = add_stain(mixed, "light")
cv2.imwrite(str(output_dir / "defect_mixed_scratch_stain.png"), mixed)
print("Created: defect_mixed_scratch_stain.png")

base = create_base_texture(256, "fabric")
mixed = add_hole(base, "heavy")
mixed = add_crack(mixed, "medium")
cv2.imwrite(str(output_dir / "defect_mixed_hole_crack.png"), mixed)
print("Created: defect_mixed_hole_crack.png")

print(f"\n✅ Generated {len(list(output_dir.glob('*.png')))} test images in: {output_dir.absolute()}")
print("\nYou can now upload these images to the Inspection page at http://localhost:5173/inspection")
