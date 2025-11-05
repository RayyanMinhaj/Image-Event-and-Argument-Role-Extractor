import os
import json
from tqdm import tqdm

DATA_DIR = './data/imsitu'

def split_annotations(split_name):
    """
    Reads the large annotation JSON file for a split (e.g., 'train') and
    splits it into individual JSON files for each image.
    """
    
    # Define paths
    input_json_path = os.path.join(DATA_DIR, f'{split_name}.json')
    output_dir = os.path.join(DATA_DIR, f'{split_name}_annotations')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {input_json_path}...")
    with open(input_json_path, 'r') as f:
        full_annotations = json.load(f)
    print("Loading complete.")
        
    print(f"Splitting annotations for '{split_name}' split into {output_dir}...")
    for image_filename, annotation_data in tqdm(full_annotations.items(), desc=f"Processing {split_name}"):
        # The image filename might be 'verb_123.jpg'. The output will be 'verb_123.json'.
        output_filename = os.path.splitext(image_filename)[0] + '.json'
        output_filepath = os.path.join(output_dir, output_filename)
        
        with open(output_filepath, 'w') as f:
            json.dump(annotation_data, f)
            
    print(f"Successfully split {len(full_annotations)} annotations for '{split_name}'.")

if __name__ == '__main__':
    split_annotations('train')
    split_annotations('dev')
    split_annotations('test')
    print("\nAll annotation files have been split.")
