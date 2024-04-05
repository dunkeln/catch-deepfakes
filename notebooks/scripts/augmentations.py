import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=.5),
    A.RandomBrightnessContrast(p=.5)
])

augmented = lambda x: transform(image=x)['image']