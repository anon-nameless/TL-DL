import rasterio.errors
import torch
import os
import rasterio
import warnings
from torch.utils.data import Dataset
from glob import glob


class LandslideDataset(Dataset):
    def __init__(self, data_root, data_list=None, transforms=None):
        """
        data_root: collect all "img" sub folders as input images,
        and all "mask" sub folders as targets.
        data_list: contain all basename of imgs&masks.
        transforms: a function or callable to process images or targets.
        """
        super(LandslideDataset, self).__init__()
        assert os.path.exists(data_root), f"path'{data_root}' does not exists."
        self.transforms = transforms
        if data_list is None:
            img_list = glob(os.path.join(data_root, "**/img/*.[tT][Ii][Ff]")) +\
                glob(os.path.join(data_root, "img/*.[tT][Ii][Ff]"))
            mask_list = glob(os.path.join(data_root, "**/mask/*.[tT][Ii][Ff]")) +\
                glob(os.path.join(data_root, "mask/*.[tT][Ii][Ff]"))

        assert len(img_list) == len(mask_list)
        for i, m in zip(img_list, mask_list):
            assert os.path.basename(i).lower() == os.path.basename(m).lower(), (
                os.path.basename(i), os.path.basename(m))
        self.img_list = img_list
        self.mask_list = mask_list
        self.transforms = transforms
        warnings.simplefilter("ignore",
                              category=rasterio.errors.NotGeoreferencedWarning)

    def __getitem__(self, index):
        img = rasterio.open(self.img_list[index]).read()  # (3,512,512)
        mask = rasterio.open(self.mask_list[index]).read()  # (1, 512,512)
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":
    dataset = LandslideDataset("../CAS mini/test")
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=5,
                                             num_workers=1,
                                             shuffle=False,
                                             pin_memory=False,
                                             drop_last=True)

    for c, (i, m) in enumerate(dataloader):
        if i.shape[-2:] != torch.Size([512, 512]):
            print(c, i.shape, m.shape)
        if m.shape[-2:] != torch.Size([512, 512]):
            print(c, i.shape, m.shape)
        if i.shape[-2:] != m.shape[-2:]:
            print(c, i.shape, m.shape)
        print(i.shape, m.shape)
        break
