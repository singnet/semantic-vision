import pandas as pd
import vqa_data_parser as vqp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os, sys, time, re


pathVocabFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/yesno_predadj_words.txt'
pathFeaturesTrainParsed = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/train2014_parsed_features'
pathFeaturesValParsed = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/val2014_parsed_features'
IMAGE_ID_FIELD_NAME = 'imageId'


pathDataTrainFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/train2014_questions_parsed.txt'
FILE_PREFIX = 'COCO_train2014_'

pathDataValFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/val2014_questions_parsed.txt'
# FILE_PREFIX = 'COCO_val2014_'

pathSaveModel = './saved_models'

input_size = 2048
nBBox = 36
featOffset = 10
nEpoch = 100
learning_rate = 1e-3

# Load vocabulary
vocab = []
with open(pathVocabFile, 'r') as filehandle:
    vocab = [current_place.rstrip() for current_place in filehandle.readlines()]

Nnets = len(vocab)

def getWords(groundedFormula):
    words = re.split(r', ', groundedFormula[groundedFormula.find("(") + 1:groundedFormula.find(")")])
    return words


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NetsVocab(nn.Module):
    def __init__(self):
        super(NetsVocab, self).__init__()
        self.models = nn.ModuleList([nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
            ).to(device) for i in range(Nnets)])

    def feed_forward(self, x, idx):
        output = torch.ones(size=(nBBox,1)).to(device)
        for k in idx:
            logits = self.models[k](x)
            # logits = model(x).view(-1)
            predict = F.sigmoid(logits)
            output = torch.mul(output, predict)
        return output

    def getParams(self, idx):
        params=[]
        for i in idx:
            params.append({'params': self.models[i].parameters()})
        return params


nets = NetsVocab()


# pathQuestFile_tab = '~/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/parsed_questions_tab.txt'
# df_tab = pd.read_csv(pathQuestFile_tab, header=0, sep='\t',  low_memory=False)
# df_quest = df_tab.loc[(df_tab['questionType'] == 'yes/no') & (df_tab['relexFormula'] == '_predadj(A, B)')]

# Load bbox features
df = pd.read_csv(pathDataTrainFile, header=0, sep='\s*\::',  engine='python')
# df = pd.read_csv(pathDataValFile, header=0, sep='\s*\::',  engine='python')
df_quest = df.loc[(df['questionType'] == 'yes/no') & (df['relexFormula'] == '_predadj(A, B)')]
df_quest = df_quest.sort_values(['imageId'], ascending=[True])
df_quest = df_quest.reset_index(drop=True)

imgIdList = df_quest[IMAGE_ID_FIELD_NAME].tolist()


# !! FOR DEBUG LOAD ONLY 1% OF DATA !!! HARDCODED INSIDE vpq.load_parsed_features !!!!
data_feat =  vqp.load_parsed_features(pathFeaturesTrainParsed, imgIdList, filePrefix=FILE_PREFIX)
# data_feat =  vqp.load_parsed_features(pathFeaturesValParsed, imgIdList, filePrefix=FILE_PREFIX)

# df.to_csv('parsed_yes_no_predadj.tsv', sep='\t', header=True, index=None)

# Get list with binary answers yes = 1, no = 0
nQuest = df_quest.shape[0]
ansList = df_quest['answer'].tolist()
for i in range(nQuest):
    if ansList[i] == 'yes':
        ansList[i] = 1
    else:
        ansList[i] = 0

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!!
nQuest = len(data_feat)
min_loss = 1e8
for e in range(nEpoch):
    mean_loss = 0.
    for i in range(nQuest):
        idx = []
        words = getWords(df_quest.loc[i, 'groundedFormula'])
        nWords = len(words)
        for w in words:
            try:
                idx.append( vocab.index(w) )
            except ValueError:
                continue

        # get img bbox features
        img_id = df_quest.loc[i, 'imageId']
        ind = imgIdList.index(img_id)
        f_input = data_feat[ind][1][:, featOffset:]
        inputs = Variable(torch.Tensor(f_input)).to(device)


        # Feed each bbox feature in the batch (36) to selected nets and multiply output probabilities
        output = nets.feed_forward(inputs, idx)

        ans = torch.from_numpy(np.asarray(ansList[i], dtype=np.float32)).to(device)
        if (ans > 0):
            output = torch.max(output).to(device)
        else:
            ans = torch.zeros((nBBox,1)).to(device)

        loss = F.binary_cross_entropy(output, ans)

        params = nets.getParams(idx)
        optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rez_loss = loss.data.cpu().numpy()
        mean_loss += rez_loss

        sys.stdout.write("\r \r Training:\tepoch: {0}/{1}\tquestion: {2}/{3}\tloss: {4}".format( e, nEpoch, i, nQuest, rez_loss) )
        sys.stdout.flush()
        time.sleep(0.01)

    mean_loss /= float(nQuest)
    print("Epoch {0}\tMean loss {1}\n".format( e, mean_loss))

    if (mean_loss < min_loss):
        state = {'epoch': e, 'state_dict': nets.state_dict(), 'optimizer' : optimizer.state_dict(), 'mean_loss' : mean_loss }
        filename = pathSaveModel + '/model.pth.tar'
        torch.save(state, filename)

print("Training is done!")


# Loading/Resuming from the dictionary
'''
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
'''