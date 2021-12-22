import torch
import torchvision.transforms as T
import torch.nn.init
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 랜덤 시드 고정
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

learning_rate = 1e-3
training_epochs = 500
batch_size = 100

custom_train_dataset = datasets.ImageFolder(root="../data/train/", transform=T.Compose([T.Grayscale(1),
                                                                                           #T.RandomInvert(1),
                                                                                           T.ToTensor()
                                                                                           ]))

custom_test_dataset = datasets.ImageFolder(root="../data/test/", transform=T.Compose([T.Grayscale(1),
                                                                                     #T.RandomInvert(1),
                                                                                     T.ToTensor()
                                                                                     ]))

train_loader = DataLoader(dataset=custom_train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True)

test_loader = DataLoader(dataset=custom_test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False)


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # L1 ImgIn shape=(?, 28, 28, 1)
        # Conv -> (?, 28, 28, 128)
        # Pool -> (?, 14, 14, 128)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            #nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # L2 ImgIn shape=(?, 14, 14, 128)
        # Conv ->(?, 14, 14, 256)
        # Pool ->(?, 7, 7, 256)
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # L3 ImgIn shape=(?, 7, 7, 256)
        # Conv ->(?, 7, 7, 512)
        # Pool ->(?, 4, 4, 512)
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        # L4 ImgIn shape=(?, 4, 4, 512)
        # Conv ->(?, 4, 4, 26)
        # Pool ->(?, 1, 1, 26)
        self.gap = nn.Sequential(
            nn.Conv2d(512, 26, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(26),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = out.view(-1, 26)
        return out

model = CNN().to(device)
#model = torch.load("../weight/p201710885.pth")

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=15)

def accuracyCalculation(model, test_loader):
    Accuracy = 0
    model = model.eval()

    with torch.no_grad():
        for num, data in tqdm(enumerate(test_loader)):
            X_test, Y_test = data
            X_test = X_test.to(device)
            Y_test = Y_test.to(device)

            prediction = model(X_test)
            correct_prediction = torch.argmax(prediction, 1) == Y_test
            accuracy = correct_prediction.float().mean()
            Accuracy += accuracy / len(test_loader)

    return Accuracy

bestepoch = 0
total_batch = len(train_loader)
bestaccuracy = -1
for epoch in range(training_epochs):
    avg_cost = 0
    model.train()

    for num, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        X, Y = data
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    accuracy = accuracyCalculation(model, test_loader)

    if accuracy > bestaccuracy:
        torch.save(model, f"../weight/model.pth")
        bestaccuracy = accuracy
        bestepoch = epoch


    scheduler.step(accuracy)
    print('[Epoch: {:>4}]\ncost = {:>.9}'.format(epoch + 1, avg_cost))
    print(f"now Accuracy: {accuracy}")
    print(f"Best Epoch: {bestepoch}, Best Accuracy: {bestaccuracy}")

print("Learning Finished")
