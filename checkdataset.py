

import os

dataset_path = r"C:\Users\renuk\Downloads\crop\test"  # Update if needed

class_counts = {}
for class_name in sorted(os.listdir(dataset_path)):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):  # Only count directories (classes)
        class_counts[class_name] = len(os.listdir(class_dir))

# Print class-wise image count
for class_name, count in class_counts.items():
    print(f"{class_name}: {count} images")

# Total images
total_images = sum(class_counts.values())
print(f"\nTotal Images: {total_images}")


