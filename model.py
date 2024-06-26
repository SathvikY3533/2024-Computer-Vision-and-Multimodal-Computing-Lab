import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm

class Conv_Block(nn.Module):
    def __init__(self, Cin, Cout, k):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout, k, padding=1)
        self.conv2 = nn.Conv2d(Cout, Cout, k, padding=1)
        self.conv3 = nn.Conv2d(Cout, Cout, k, padding=1)
        self.batchNorm = nn.BatchNorm2d(Cout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        #x = self.batchNorm(x)
        x = self.relu(self.conv2(x))
        #x = self.batchNorm(x)
        x = self.relu(self.conv3(x))
        #x = self.batchNorm(x)
        return x

class Conv_Block_Last(nn.Module):
    def __init__(self, Cin, Cout, k):
        super(Conv_Block_Last, self).__init__()
        self.conv = nn.Conv2d(Cin, Cout, k, padding=k//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x

class MP2D(nn.Module):
    def __init__(self, k, stride):
        super(MP2D, self).__init__()
        self.pool = nn.MaxPool2d(k, stride=stride)

    def forward(self, x):
        return self.pool(x)

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, device='gpu', **kwargs):
        super(model, self).__init__()

        self.visualModel = None
        self.audioModel = None
        self.fusionModel = None
        self.fcModel = None

        self.device = ("cuda" if torch.cuda.is_available() else 'cpu')

        self.createVisualModel()
        self.createAudioModel()
        self.createFusionModel()
        self.createFCModel()
        
        self.visualModel = self.visualModel.to(self.device)
        self.audioModel = self.audioModel.to(self.device)
        self.fcModel = self.fcModel.to(self.device)
        
        self.optim = torch.optim.Adam(self.parameters(), lr = lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size = 1, gamma=lrDecay)
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def createVisualModel(self):
        self.visualModel = nn.Sequential(
            Conv_Block(1, 32, 3),
            MP2D(2, (2, 2)),
            Conv_Block(32, 64, 3),
            MP2D(2, (2, 2)),
            Conv_Block(64, 64, 3),
            MP2D(2, (2, 2)),
            Conv_Block_Last(64, 128, 3)
        )

    def createAudioModel(self):
        self.audioModel = nn.Sequential(
            Conv_Block(1, 32, 3),
            MP2D(2, (2, 1)),
            Conv_Block(32, 64, 3),
            MP2D(2, (2, 1)),
            Conv_Block(64, 64, 3),
            MP2D(2, (2, 1)),
            Conv_Block(64, 64, 3),
            MP2D(2, (2, 1)),
            Conv_Block_Last(64, 128, 3)
        )


    def createFusionModel(self):
        pass

    def createFCModel(self):
        self.fcModel = nn.Sequential(
            nn.Flatten(),
            nn.Linear(41472, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            #nn.Dropout(p=0.3),
            nn.Linear(128, 2)
        )
    
    def train_network(self, loader, epoch, **kwargs):
        self.train()
        self.scheduler.step(epoch-1)
        lr = self.optim.param_groups[0]['lr']
        index, top1, loss = 0, 0, 0
        for num, (audioFeatures, visualFeatures, labels) in enumerate(loader, start=1):
                self.zero_grad()

                # print('audioFeatures shape: ', audioFeatures.shape)
                # print('visualFeatures shape: ', visualFeatures.shape)
                # print('labels shape: ', labels.shape)
                
                audioFeatures = torch.unsqueeze(audioFeatures, dim=1)  
                # print('audioFeatures after unsqueeze: ', audioFeatures.shape)            
                
                audioFeatures = audioFeatures.to(self.device)
                visualFeatures = visualFeatures.to(self.device)
                labels = labels.squeeze().to(self.device)
                                
                audioEmbed = self.audioModel(audioFeatures)
                # print('audio embed shape: ', audioEmbed.shape)
                visualEmbed = self.visualModel(visualFeatures)
                visualEmbed = F.interpolate(visualEmbed, size=(18, 9), mode='bilinear', align_corners=False)
                # print('visual embed shape: ', visualEmbed.shape)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
                # print('avfusion shape: ', avfusion.shape)
                
                fcOutput = self.fcModel(avfusion)
                # print('fc output shape: ', fcOutput.shape)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                self.optim.zero_grad()
                nloss.backward()
                self.optim.step()
                
                loss += nloss.detach().cpu().numpy()
                
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
                " [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
                " Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), 100 * (top1/index)))
                sys.stderr.flush()  
        sys.stdout.write("\n")
        
        return loss/num, lr
        
    def evaluate_network(self, loader, **kwargs):
        self.eval()
        predScores = []
        
        loss, top1, index, numBatches = 0, 0, 0, 0
        
        for audioFeatures, visualFeatures, labels in tqdm.tqdm(loader):
            
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)
            audioFeatures = audioFeatures.to(self.device)
            visualFeatures = visualFeatures.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            with torch.no_grad():
                
                audioEmbed = self.audioModel(audioFeatures)
                visualEmbed = self.visualModel(visualFeatures)
                visualEmbed = F.interpolate(visualEmbed, size=(18, 9), mode='bilinear', align_corners=False)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
            
                fcOutput = self.fcModel(avfusion)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1
                
        print('eval loss ', loss/numBatches)
        print('eval accuracy ', top1/index)
        
        return top1/index

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)
        
    def loadParameters(self, path):
        selfState = self.state_dict()
        loadedState = torch.load(path)
        for name, param in loadedState.items():
            origName = name
            if name not in selfState:
                name = name.replace("module.", "")
                if name not in selfState:
                    print("%s is not in the model."%origName)
                    continue
            if selfState[name].size() != loadedState[origName].size():
                sys.stderr.write("Wrong parameter length: %s, model: %s, loaded: %s"%(origName, selfState[name].size(), loadedState[origName].size()))
                continue
            selfState[name].copy_(param)