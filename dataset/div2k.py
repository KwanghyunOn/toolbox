import os

from .image import ImageDataset


class DIV2K(ImageDataset):
    """
    Root directory should look like

        root/
            DIV2K_train_HR/
                0000.png
                0001.png
                ...
            DIV2K_train_LR_bicubic/
                x2/
                    0000x2.png
                    0001x2.png
                    ...
                x3/
                x4/
            DIV2K_valid_HR/
                0801.png
                0802.png
                ...
            DIV2K_valid_LR_bicubic/
                x2/
                x3/
                x4/
    """

    def __init__(
        self, root, scale, train, transform=None, is_binary=True,
    ):
        self.data_len = data_len
        split = "train" if train else "valid"
        deg = "bicubic"
        hr_data_dir = os.path.join(root, f"DIV2K_{split}_HR")
        lr_data_dir = os.path.join(root, f"DIV2K_{split}_LR_{deg}", f"X{scale}_upsampled")
        data_dirs = {"hr": hr_data_dir, "lr": lr_data_dir}
        super().__init__(data_dirs, transform, is_binary, train)

    def __len__(self):
        if self.data_len is not None and self.data_len > 0:
            return self.data_len
        else:
            return super().__len__()
