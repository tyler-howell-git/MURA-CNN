from torchvision import transforms

def get_train_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #ensure remains grayscale (1 output channel)
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])

def get_val_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485],
                             std=[0.229])
    ])
