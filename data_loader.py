import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob

class ImSituDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.image_dir = os.path.join(data_dir, 'images_256')
        self.annotation_dir = os.path.join(data_dir, f'{split}_annotations') 

        # Load metadata (this is small and safe to keep in memory)
        with open(os.path.join(data_dir, 'imsitu_space.json')) as f:
            self.imsitu_space = json.load(f)
        
        self.verb_to_idx = {verb: i for i, verb in enumerate(self.imsitu_space['verbs'])}
        self.noun_to_idx = {noun: i for i, noun in enumerate(self.imsitu_space['nouns'])}

        all_roles = set()
        for verb_data in self.imsitu_space['verbs'].values():
            for role in verb_data['roles']:
                all_roles.add(role)
        self.role_to_idx = {role: i for i, role in enumerate(sorted(list(all_roles)))}
        self.num_roles = len(self.role_to_idx)
        self.num_nouns = len(self.noun_to_idx)

        # --- THE BIG CHANGE [5/11/2025] ---
        # Instead of loading a giant JSON, we just get a list of annotation file paths.
        # This uses virtually no memory.
        self.annotation_filepaths = sorted(glob.glob(os.path.join(self.annotation_dir, '*.json')))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.annotation_filepaths)

    def __getitem__(self, idx):
        # Load the annotation for a SINGLE image
        annotation_path = self.annotation_filepaths[idx]
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        # Get image name from the annotation file name
        img_json_filename = os.path.basename(annotation_path)
        img_name = os.path.splitext(img_json_filename)[0] + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load the corresponding image
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except (IOError, FileNotFoundError):
            print(f"Warning: Could not load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # Process the loaded annotation (same logic as before)
        verb_str = annotation['verb']
        verb_idx = self.verb_to_idx[verb_str]

        frame = annotation['frames'][0]
        
        noun_indices = torch.full((self.num_roles,), -1, dtype=torch.long)
        for role, noun_str in frame.items():
            if role in self.role_to_idx and noun_str in self.noun_to_idx:
                role_idx = self.role_to_idx[role]
                noun_idx = self.noun_to_idx[noun_str]
                noun_indices[role_idx] = noun_idx

        return {
            'image': image,
            'verb_idx': torch.tensor(verb_idx, dtype=torch.long),
            'noun_indices': noun_indices
        }


if __name__ == '__main__':
    data_dir = './data/imsitu'
    train_dataset = ImSituDataset(data_dir=data_dir, split='train')
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of verbs: {len(train_dataset.verb_to_idx)}")
    print(f"Number of nouns: {train_dataset.num_nouns}")
    print(f"Number of unique roles: {train_dataset.num_roles}")

    sample = train_dataset[0]
    print("\nSample shapes:")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Verb index shape: {sample['verb_idx'].shape}")
    print(f"Noun indices shape: {sample['noun_indices'].shape}")
