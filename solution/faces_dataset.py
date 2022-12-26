"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        # image should be tensor
        # label - 0 for real, 1 for fake
        # TODO: check the following if
        if index < int(self.__len__() / 2):
            label = 0
            image_path = os.path.join(self.root_path, 'real', self.real_image_names[index])

        else:
            index = index - len(self.real_image_names)
            label = 1
            image_path = os.path.join(self.root_path, 'fake', self.fake_image_names[index])

        im = Image.open(image_path)
        if self.transform is not None:
            im_tensor = self.transform(im)
        if not torch.is_tensor(im):
            transform = transforms.Compose([transforms.PILToTensor()])
            im_tensor = transform(im)

        return im_tensor, label

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.real_image_names) + len(self.fake_image_names)
