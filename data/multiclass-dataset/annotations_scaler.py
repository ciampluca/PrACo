import json, os


input_path = "./annotations.json"
output_path = "./annotations_no_scaling.json"
# load annotations in FSC format from input_path

annotations = json.load(open(input_path, "r"))


output_annotations = {}

# remove scaling factor from annotations
for img_name, img_data in annotations.items():

    target_height = img_data['H']
    target_width = img_data['W']

    points = img_data['points']
    classes = img_data['classes']

    scaled_points = []

    aspect_ratio = target_width / target_height
    
    #target_width = int(16 * round((target_height * aspect_ratio) / 16))  # Round to multiple of 16
    
    # Points are already at 384px height scale, no need to scale them
    # But we need to scale them to target dimensions if target_height != 384
    scale_factor = min(target_height, target_width) / 384.0

    for point in points:
        x, y = point
        # Scale points if target is not 384
        x_scaled = x * scale_factor
        y_scaled = y * scale_factor
        
        x_scaled = int(round(x_scaled))
        y_scaled = int(round(y_scaled))
        # Ensure coordinates are within bounds
        assert 0 <= x_scaled <= target_width, f"x_scaled {x_scaled} out of bounds for target_width {target_width}"
        assert 0 <= y_scaled <= target_height, f"y_scaled {y_scaled} out of bounds for target_height {target_height}"
        #x_int = min(target_width - 1, max(0, int(round(x_scaled))))
        #y_int = min(target_height - 1, max(0, int(round(y_scaled))))
        
        scaled_points.append([x_scaled, y_scaled])

    output_annotations[img_name] = {
        "points": scaled_points,
        "classes": classes
    }


json.dump(output_annotations, open(output_path, "w"), indent=4)