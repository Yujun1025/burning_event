import torch.nn as nn
import torch.nn.functional as F
class ClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(ClassifierMLP, self).__init__()
        
        self.last = last
        self.net = nn.Sequential(  
                   nn.Linear(input_size, int(input_size/4)),
                   nn.ReLU(),
                   nn.Dropout(p=dropout),  
                   nn.Linear(int(input_size/4), output_size))
              
        
        if last == 'logsm':
            self.last_layer = nn.LogSoftmax(dim=-1)
        elif last == 'sm':
            self.last_layer = nn.Softmax(dim=-1)
        elif last == 'tanh':
            self.last_layer = nn.Tanh()
        elif last == 'sigmoid':
            self.last_layer = nn.Sigmoid()
        elif last == 'relu':
            self.last_layer = nn.ReLU()

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)
        
        return y
     


class extractor(nn.Module):
    
    def __init__(self):
        super(extractor, self).__init__()
       
        self.conv_1 = nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24)
        self.bn_1 = nn.BatchNorm1d(16)
        self.relu_1 = nn.ReLU(inplace=True)
        self.pool_1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv_2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding='same')
        self.bn_2 = nn.BatchNorm1d(32)
        self.relu_2 = nn.ReLU(inplace=True)
        self.pool_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv_3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same')
        self.bn_3 = nn.BatchNorm1d(64)
        self.relu_3 = nn.ReLU(inplace=True)
        self.pool_3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv_4 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn_4 = nn.BatchNorm1d(64)
        self.relu_4 = nn.ReLU(inplace=True)
        self.pool_4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv_5 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding='same')
        self.bn_5 = nn.BatchNorm1d(64)
        self.relu_5 = nn.ReLU(inplace=True)
        self.pool_5 = nn.MaxPool1d(kernel_size=2, stride=2)
        

        self.fl = nn.Flatten()
        self. dropout = nn.Dropout(0.2)
    
    def forward(self, input):
        
        out = self.conv_1(input)
        out = self.bn_1(out)
        out = self.relu_1(out)
        out = self.pool_1(out)
        
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = self.relu_2(out)
        out = self.pool_2(out)
        
        out = self.conv_3(out)
        out = self.bn_3(out)
        out = self.relu_3(out)
        f = self.pool_3(out)
        
        out = self.conv_4(f)
        out = self.bn_4(out)
        out = self.relu_4(out)
        out = self.pool_4(out)
        
        # out = self.conv_5(out)
        # out = self.bn_5(out)
        # out = self.relu_5(out)
        # out = self.pool_5(out)
        
        out = self.fl(out)
        out = F.normalize(out, dim = 1)
        return out,f 

class Shared_feature_decoder(nn.Module):
    def __init__(self):
        super(Shared_feature_decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 64 * 256),
            nn.ReLU(),
            nn.Unflatten(1, (64, 256)),
            #(number of kernels, kernel size, padding) - (pooling size, pooling stride)
            nn.ConvTranspose1d(64, 64, 5, stride=2, padding=2, output_padding=1),  
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1),
        )
    def forward(self, x):
        return self.layers(x)
    

class BaseModel(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()
        
        self.G = extractor(in_channel=input_size)
        
        self.C = ClassifierMLP(512, num_classes, dropout, last=None)
        
    def forward(self, input):
        f,_ = self.G(input)
        predictions = self.C(f)
        
        if self.training:
            return predictions, f
        else:
            return predictions