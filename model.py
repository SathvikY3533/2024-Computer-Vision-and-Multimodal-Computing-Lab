import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, tqdm
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import trainer
import warnings
import torchvision.models as models
import torchvggish

warnings.filterwarnings("ignore")

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
        #print(f"After conv1: {x.shape}")
        #x = self.batchNorm(x)
        x = self.relu(self.conv2(x))
        #print(f"After conv2: {x.shape}")
        #x = self.batchNorm(x)
        x = self.relu(self.conv3(x))
        #print(f"After conv3: {x.shape}")
        #x = self.batchNorm(x)
        return x

class Conv_Block_Last(nn.Module):
    def __init__(self, Cin, Cout, k):
        super(Conv_Block_Last, self).__init__()
        self.conv = nn.Conv2d(Cin, Cout, k, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        #print(f"After last conv: {x.shape}")
        return x

class MP2D(nn.Module):
    def __init__(self, k, stride):
        super(MP2D, self).__init__()
        self.pool = nn.MaxPool2d(k, stride=stride)

    def forward(self, x):
        x = self.pool(x)
        #print(f"After MaxPool: {x.shape}")
        return x

class model(nn.Module):
    def __init__(self, lr=0.0001, lrDecay=0.95, device='gpu', visual_model=None, audio_model=None, **kwargs):
        super(model, self).__init__()

        self.visualModel = visual_model
        self.audioModel = audio_model
        self.fusionModel = None
        self.fcModel = None

        if 'enableVGG' in kwargs:
            self.enableVGG = kwargs['enableVGG']
        else:
            self.enableVGG = False

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
        if(self.enableVGG == "True"):
            vgg = models.vgg16(pretrained=True)
            vgg.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            vgg = nn.Sequential(*list(vgg.features.children()))
            self.visualModel = nn.Sequential(vgg, nn.Flatten())
        else:
            self.visualModel = nn.Sequential(
                Conv_Block(5, 32, 3),
                MP2D(2, (2, 2)),
                Conv_Block(32, 64, 3),
                MP2D(2, (2, 2)),
                Conv_Block(64, 64, 3),
                MP2D(2, (2, 2)),
                Conv_Block_Last(64, 128, 3),
                nn.Flatten()
            )

    def createAudioModel(self):
        if(self.enableVGG == "True"):
            self.vggish = torchvggish.vggish()
            self.vggish.features[2] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
            self.vggish.features[5] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
            self.vggish.features[10] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
            self.vggish.features[15] = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
            self.vggish = nn.Sequential(*list(self.vggish.features.children()), nn.Flatten())
            self.audioModel = nn.Sequential(self.vggish)
        else:
            self.audioModel = nn.Sequential(
                Conv_Block(1, 32, 3),
                MP2D(2, (2, 1)),
                Conv_Block(32, 64, 3),
                MP2D(2, (2, 1)),
                Conv_Block(64, 64, 3),
                MP2D(2, (2, 1)),
                Conv_Block(64, 64, 3),
                MP2D(2, (2, 2)),
                Conv_Block_Last(64, 128, 3),
                nn.Flatten()
            )


    def createFusionModel(self):
        pass

    def createFCModel(self):
        i = 2641920
        self.fcModel = nn.Sequential(
            nn.Linear(i, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
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
        
        all_labels = []
        all_preds = []
        
        loss, top1, index, numBatches = 0, 0, 0, 0
        
        for audioFeatures, visualFeatures, labels in tqdm.tqdm(loader):
            
            audioFeatures = torch.unsqueeze(audioFeatures, dim=1)
            audioFeatures = audioFeatures.to(self.device)
            visualFeatures = visualFeatures.to(self.device)
            labels = labels.squeeze().to(self.device)
            
            with torch.no_grad():
                
                audioEmbed = self.audioModel(audioFeatures)
                visualEmbed = self.visualModel(visualFeatures)
                
                avfusion = torch.cat((audioEmbed, visualEmbed), dim=1)
            
                fcOutput = self.fcModel(avfusion)
                
                nloss = self.loss_fn(fcOutput, labels)
                
                loss += nloss.detach().cpu().numpy()
                top1 += (fcOutput.argmax(1) == labels).type(torch.float).sum().item()
                index += len(labels)
                numBatches += 1
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(fcOutput[:, 1].cpu().numpy())  # Assuming positive class is at index 1
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)
        mAP = average_precision_score(all_labels, all_preds)
        
        print('eval loss ', loss/numBatches)
        print('eval accuracy ', top1/index)
        print('eval mAP ', mAP)
        
        accuracy = top1 / index
        
        return mAP, accuracy

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
