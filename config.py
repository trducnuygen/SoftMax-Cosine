from src.data import get_dataloader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_data(data_path, mode, batch_size, num_workers=8, shuffle=False, pin_memory=True):
    # if mode == "ImageNet_small":
    if mode == "ImageNet":
        train_dir = f"{data_path}/train"
        train_loader, train_dataset = get_dataloader(train_dir, batch_size, return_dataset=True, shuffle=shuffle, num_workers=num_workers)

        test_dir = f"{data_path}/val"
        val_loader, val_dataset = get_dataloader(test_dir, batch_size, return_dataset=True, num_workers=num_workers, shuffle=shuffle)

        return train_loader, val_loader, train_dataset, val_dataset
    
    if mode == "Places365":
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        train_set = datasets.Places365(root=data_path, split="train-standard",
                                        small=True, download=False, transform=trans)
        val_set = datasets.Places365(root=data_path, split="val",
                                        small=True, download=False, transform=trans)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

        return train_loader, val_loader, train_set, val_set
