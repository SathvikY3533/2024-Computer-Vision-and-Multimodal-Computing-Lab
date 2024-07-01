import os, glob, time
import argparse
import torch
from model import *
from dataLoader_Image_audio import train_loader, val_loader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import torchvision.models as models

def parser():
    args = argparse.ArgumentParser(description="ASD Trainer")

    args.add_argument("--lr", type=float, default=0.0001, help="Learning Rate")
    args.add_argument('--lrDecay', type=float, default=0.95, help='Learning rate decay rate')
    args.add_argument('--maxEpoch', type=int, default=10, help='Maximum number of epochs')
    args.add_argument('--testInterval', type=int, default=1, help='Test and save every [testInterval] epochs')
    args.add_argument('--batchSize', type=int, default=128, help='Dynamic batch size, default is 500 frames.')
    args.add_argument('--nDataLoaderThread', type=int, default=4, help='Number of loader threads')
    args.add_argument('--datasetPath', type=str, default="/Users/sathvikyechuri/Downloads/AVDIAR_ASD_FTLim", help='Path to the ASD Dataset')
    args.add_argument('--loadAudioSeconds', type=float, default=3, help='Number of seconds of audio to load for each training sample')
    args.add_argument('--loadNumImages', type=int, default=1, help='Number of images to load for each training sample')
    args.add_argument('--savePath', type=str, default="exps/exp1")
    args.add_argument('--evalDataType', type=str, default="val", help='The dataset for evaluation, val or test')
    args.add_argument('--evaluation', dest='evaluation', action='store_true', help='Only do evaluation')
    args.add_argument('--eval_model_path', type=str, default="path not specified", help="model path for evaluation")
    args.add_argument('--enableVGG', type=str, default="False", help="use VGG model? True or False")

    args = args.parse_args()

    return args

def main(args):
    loader = train_loader(trialFileName=os.path.join(args.datasetPath, 'csv/train_loader.csv'), 
                          audioPath=os.path.join(args.datasetPath , 'clips_audios/'), 
                          visualPath=os.path.join(args.datasetPath, 'clips_videos/train'), 
                          **vars(args))
    trainLoader = torch.utils.data.DataLoader(loader, batch_size=args.batchSize, shuffle=True, num_workers=args.nDataLoaderThread)

    loader = val_loader(trialFileName=os.path.join(args.datasetPath, 'csv/val_loader.csv'), 
                        audioPath=os.path.join(args.datasetPath , 'clips_audios'), 
                        visualPath=os.path.join(args.datasetPath, 'clips_videos', args.evalDataType), 
                        **vars(args))
    valLoader = torch.utils.data.DataLoader(loader, batch_size=args.batchSize, shuffle=False, num_workers=4)
    if args.evaluation == True:
        s = model(**vars(args))

        if args.eval_model_path == "path not specified":
            print('Evaluation model parameters path has not been specified')
            quit()
        
        s.loadParameters(args.eval_model_path)
        print("Parameters loaded from path ", args.eval_model_path)
        mAP, accuracy = s.evaluate_network(loader=valLoader, **vars(args))
        print("mAP %2.2f%%, Accuracy %2.2f%%" % (mAP, accuracy))
        quit() 

    args.modelSavePath = os.path.join(args.savePath, 'model')
    os.makedirs(args.modelSavePath, exist_ok=True)
    args.scoreSavePath = os.path.join(args.savePath, 'score.txt')
    modelfiles = glob.glob('%s/model_0*.model' % args.modelSavePath)
    modelfiles.sort()
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!" % modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = model(epoch=epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = model(epoch=epoch, **vars(args))

    mAPs = []
    losses = []
    accuracies = []
    scoreFile = open(args.scoreSavePath, "a+")
    bestmAP = 0
    while True:
        loss, lr = s.train_network(epoch=epoch, loader=trainLoader, **vars(args))
        losses.append(loss)
        
        if epoch % args.testInterval == 0:        
            mAP, accuracy = s.evaluate_network(epoch=epoch, loader=valLoader, **vars(args))
            mAPs.append(mAP)
            accuracies.append(accuracy)
            if mAP > bestmAP:
                bestmAP = mAP
                s.saveParameters(args.modelSavePath + "/best.model")
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp}, {epoch} epoch, mAP {mAP:.2f}%, Accuracy {accuracy:.2f}%, bestmAP {max(mAPs):.2f}%")
            scoreFile.write(f"{timestamp}, {epoch} epoch, LR {lr:.6f}, LOSS {loss:.6f}, mAP {mAP:.2f}%, Accuracy {accuracy:.2f}%, bestmAP {max(mAPs):.2f}%\n")
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            break

        epoch += 1

    scoreFile.close()

#timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    # # Plotting the loss, mAP, and accuracy
    # epochs = list(range(1, epoch + 1))
    # plt.figure()
    # plt.plot(epochs, losses, label='Loss')
    # plt.plot(epochs, mAPs, label='mAP')
    # plt.plot(epochs, accuracies, label='Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Value')
    # plt.legend()
    # plt.title('Training Loss, mAP, and Accuracy over Epochs')
    # plt.show()

if __name__ == "__main__":
    # vgg = models.vgg16(pretrained=True)
    # print(vgg)
    # sys.exit()
    args = parser()
    main(args)
