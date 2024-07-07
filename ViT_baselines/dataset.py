from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
from PIL import Image


class Uni_Dataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            root,
            train=False,
            val=False,
            test=False,
            transforms=None,
            binary=False,
            data_percent=1
    ):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age

        """
        if train:
            self.df = df[df['split'] == 'train']
            half_rows = int(len(self.df) * data_percent)
            self.df = self.df.head(half_rows)
        elif val:
            self.df = df[df['split'] == 'val']
        elif test:
            self.df = df[df['split'] == 'test']
        self.transforms = transforms
        self.root = root
        self.binary = binary

    # import torchsnooper
    # @torchsnooper.snoop()
    def __getitem__(self, index):
        filename = self.df.iloc[index]['image']
        im_path = str(self.root) + str(filename)
        # Use PIL to load the image directly in RGB format
        try:
            x = Image.open(im_path).convert('RGB')
        except IOError:
            print('Error opening file:', im_path)
            x = None  # Or handle the error as appropriate for your application

        # Apply transformations if any
        if x is not None and self.transforms:
            x = self.transforms(x)
            if self.binary:
                y = self.df.iloc[index]['binary_label']
            else:
                y = self.df.iloc[index]['label']
        return x, y

    def __len__(self):
        return len(self.df)

    def count_label(self):
        count_one = 0
        count_two = 0
        for label in self.df['binary_label']:
            if label == 0:
                count_one += 1
            elif label == 1:
                count_two += 1
        return count_one, count_two
