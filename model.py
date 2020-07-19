def pick_activation(string): #default is relu
        if string == 'relu':
            return nn.ReLU()
        if string == 'tanh':
            return nn.Tanh()

class CNN_Classifier(nn.Module):
    def __init__(self, activation='relu'):
        super(CNN_Classifier, self).__init__()
        #input is 32x32x3 (W, H, C)
        self.activation = nn.ReLU()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        #result is 16x16x16
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        #result is 8x8x32

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        #result is 4x4x64

        #ff layer
        self.dp1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4*4*64, 512)
        self.bn4 = nn.BatchNorm1d(512)

        self.dp2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        #input coming in 32x32x3

        x = self.conv1(x)
        x = self.activation(x)
        x = self.bn1(x)
        #result 16x16x16

        x = self.conv2(x)
        x = self.activation(x)
        x = self.bn2(x)
        #result 8x8x32

        x = self.conv3(x)
        x = self.activation(x)
        x = self.bn3(x)
        #result = 4x4x64

        #reshape 3D tensor in a "1D" (2D tensor acting like a 1D tensor)
        x = x.view(x.size(0), -1)
        #resulting tensor will be [4x4x64] to feed through network

        x = self.dp1(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn4(x)

        x = self.dp2(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn5(x)

        x = self.fc3(x)

        return x
