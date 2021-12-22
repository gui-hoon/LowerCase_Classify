import torchvision.transforms as T
import torch.nn.init
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import nn
import time

start = time.perf_counter()

class CNN(torch.nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()
        self.layer1 = self.conv1(1, 128) # (_, 14, 14, 128)
        self.layer2 = self.conv2(128, 256) # (_, 7, 7, 256)
        self.layer3 = self.conv3(256, 512) # (_, 4, 4, 512)
        self.gap = self.global_avg_pool(512, 26) # (_, 1, 1, 26)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.gap(out)
        out = out.view(-1, 26)
        return out


device = "cuda" if torch.cuda.is_available() else "cpu"

# 랜덤 시드 고정
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

batch_size = 100
custom_test_dataset = datasets.ImageFolder(root="../data/test/", transform=T.Compose([T.ToTensor(),
                                                                                    #T.Resize(28),
                                                                                    #T.RandomInvert(1),
                                                                                    T.Grayscale(1)]))

test_loader = DataLoader(dataset=custom_test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False)

# 학습을 진행하지 않을 것이므로 torch.no_grad()
Accuracy = 0
#model = torch.load("../weight/3kernel_Gap_v2.pth")
model = torch.load("../weight/model.pth")
model = model.eval()

with torch.no_grad():
    for num, data in enumerate(test_loader):
        X_test, Y_test = data
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        Accuracy += accuracy / len(test_loader)

    print('Accuracy:', Accuracy)

print(f"time: {(time.perf_counter() - start)/len(custom_test_dataset)}")
