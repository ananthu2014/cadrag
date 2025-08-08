import pandas as pd
import os
from PIL import Image
from src import clip
import torch
from torch.utils.data import Dataset, DataLoader

class CSTBIR_dataset(Dataset):
    def __init__(self, data_path, image_files_path, sketch_files_path, split, preprocess, return_paths=False):

        self.data = pd.read_csv(data_path)
        if isinstance(split, list):
            self.data = self.data[self.data['split'].isin(split)]
        else:
            self.data = self.data[self.data['split'] == split]
        self.texts = self.data['text'].to_list()
        self.images = self.data['image_filename'].to_list()
        self.images = [os.path.join(image_files_path, image_filename) for image_filename in self.images]
        self.sketches = self.data['sketch_filename'].to_list()
        self.sketches = [os.path.join(sketch_files_path, sketch_filename) for sketch_filename in self.sketches]
        self.n_samples = len(self.images)
        self.preprocess = preprocess

        self.image2sampleidx = {}
        for idx in range(len(self.images)):
            image_filename = self.images[idx]
            if image_filename not in self.image2sampleidx:
                self.image2sampleidx[image_filename] = []
            self.image2sampleidx[image_filename].append(idx)

        self.image2text = {}
        self.text2image = {}
        for idx in range(self.n_samples):
            image_filename = self.images[idx]
            text = self.texts[idx]
            if image_filename not in self.image2text:
                self.image2text[image_filename] = []
            if text not in self.text2image:
                self.text2image[text] = []
            self.image2text[image_filename].append(text)
            self.text2image[text].append(image_filename)

        self.unique_images = list(self.image2sampleidx.keys())
        assert len(self.texts) == len(self.images) == len(self.sketches)
        self.return_paths = return_paths
        self.indices = self.data.index.tolist()

        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
            image = self.preprocess(Image.open(self.images[idx]))
            sketch = self.preprocess(Image.open(self.sketches[idx]))
            index = self.indices[idx]

            orig_text = str(self.texts[idx])

            if pd.isna(orig_text) or orig_text.strip() == '':  # Better nan checking
                text = '[PAD]'  # Use a special token instead of empty string
            else:
                text = clip.tokenize(orig_text, truncate=True)[0]

            
            if self.return_paths:
                
                return image, sketch, text, self.images[idx], self.sketches[idx], index
            else:
                return image, sketch, text


def get_dataloader(data_path, image_files_path, sketch_files_path, batch_size, split, preprocess, num_workers=4, shuffle=True, return_paths=False):
    dataset = CSTBIR_dataset(data_path, image_files_path, sketch_files_path, split, preprocess, return_paths)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)