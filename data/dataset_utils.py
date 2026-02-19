import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
from torchvision import transforms

def get_pneumonia_data(batch_size=64):
    data_flag = 'pneumoniamnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5]),
    ])
    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    val_dataset = DataClass(split='val', transform=test_transform, download=True)
    test_dataset = DataClass(split='test', transform=test_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
