import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader

batch_size = 100

custom_train_dataset = datasets.ImageFolder(root="./datasetSmall/train/", transform = T.Compose([T.ToTensor(),
                                                                                          T.Grayscale(1)]))

custom_test_dataset = datasets.ImageFolder(root="./datasetSmall/validation/", transform=T.Compose([
                                                                                    T.Grayscale(1),
                                                                                    T.ToTensor()]))

train_loader = DataLoader(dataset=custom_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

test_loader = DataLoader(dataset=custom_test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          drop_last=False)

print(f"train size: {len(custom_train_dataset)}")
print(f"test size: {len(custom_test_dataset)}")
print("train_dataset_classes_names = ", custom_train_dataset.class_to_idx)
print("num classes = ",len(custom_train_dataset.class_to_idx))

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.size())