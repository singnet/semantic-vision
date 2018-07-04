import pandas as pd
import vqa_data_parser as vqp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os, sys, time, re
import math

pathVocabFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/yesno_predadj_words.txt'

pathImgs = '/home/mvp/Desktop/SingularityNET/my_exp/Attention-on-Attention-for-VQA/data/val2014'
pathFeaturesParsed = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/val2014_parsed_features'
pathQuestFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/val2014_questions_parsed.txt'
FILE_PREFIX = 'COCO_val2014_'

# pathImgs = '/home/mvp/Desktop/SingularityNET/my_exp/Attention-on-Attention-for-VQA/data/train2014'
# pathFeaturesParsed = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/train2014_parsed_features'
# pathQuestFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/train2014_questions_parsed.txt'
# FILE_PREFIX = 'COCO_train2014_'


IMAGE_ID_FIELD_NAME = 'imageId'
id_len = 12


pathSaveModel = './saved_models_00'


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
checkpoint = torch.load(pathSaveModel + '/model.pth.tar')
mean_loss = checkpoint['mean_loss']
ep = checkpoint['epoch']
nets.load_state_dict(checkpoint['state_dict'])
print("Mean loss value: {} (epoch {})" .format(mean_loss, checkpoint['epoch']))


# pathQuestFile_tab = '~/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/parsed_questions_tab.txt'
# df_tab = pd.read_csv(pathQuestFile_tab, header=0, sep='\t',  low_memory=False)
# df_quest = df_tab.loc[(df_tab['questionType'] == 'yes/no') & (df_tab['relexFormula'] == '_predadj(A, B)')]

# Load bbox features
#df = pd.read_csv(pathDataTrainFile, header=0, sep='\s*\::',  engine='python')
df = pd.read_csv(pathQuestFile, header=0, sep='\s*\::',  engine='python')
df_quest = df.loc[(df['questionType'] == 'yes/no') & (df['relexFormula'] == '_predadj(A, B)')]
df_quest = df_quest.sort_values(['imageId'], ascending=[True])
df_quest = df_quest.reset_index(drop=True)

imgIdList = df_quest[IMAGE_ID_FIELD_NAME].tolist()
# Drop duplicates and sort
imgIDSet = sorted(set(imgIdList))

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!! HARDCODED INSIDE vpq.load_parsed_features !!!!
data_feat =  vqp.load_parsed_features(pathFeaturesParsed, imgIdList, filePrefix=FILE_PREFIX, reduce_set=True)

# df.to_csv('parsed_yes_no_predadj.tsv', sep='\t', header=True, index=None)

# Get list with binary answers yes = 1, no = 0
nQuest = df_quest.shape[0]
questList = df_quest['question'].tolist()
ansList = df_quest['answer'].tolist()
ansListBin = []
for i in range(nQuest):
    if ansList[i] == 'yes':
        ansListBin.append(1)
    else:
        ansListBin.append(0)

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!!
nQuest = len(data_feat)
min_loss = 1e8

fileLog = open("log_val.txt", 'w')

file = open("eval_results.html", 'w')
file.write('<html>\n'
           '<head>\n'
           '<style>\n'
           '.one{width: 700px;height: 700px;position: relative;}\n'
           '.two{width: 700px;height: 700px;position: absolute;}\n'
           '</style>\n'
           '</head>\n'
           '<body>\t<table>\n')

score = 0.
nQuestValid = 0
ans_stat = []
num_yes_gt = 0
num_yes_pred = 0
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
    filePath = pathImgs + '/' + FILE_PREFIX
    nZeros = int((id_len - 1) - math.floor(math.log10(img_id)))
    for _ in range(0, nZeros):
        filePath = filePath + '0'

    filePath = filePath + str(img_id) + '.jpg'


    ind = imgIdList.index(img_id)
    bboxes = data_feat[ind][1][:, 0:4]
    f_input = data_feat[ind][1][:, featOffset:]
    inputs = Variable(torch.Tensor(f_input)).to(device)


    # Feed each bbox feature in the batch (36) to selected nets and multiply output probabilities
    output = nets.feed_forward(inputs, idx)

    # ans = torch.from_numpy(np.asarray(ansList[i], dtype=np.float32)).to(device)
    ans = np.asarray(ansListBin[i], dtype=np.float32)
    output_max, idx_max = torch.max(output, 0)
    imax = idx_max.data.cpu().numpy()

    output_max = np.round(output_max.data.cpu().numpy())[0]

    file.write('\t<tr><td><div class="one"><div class="two"><img src="' + filePath + '"/></div>')
    file.write('')

    for j in range(bboxes.shape[0]):
    # for j in range(1):
        x = int(bboxes[j][0])
        y = int(bboxes[j][1])
        width = int(bboxes[j][2])
        height = int(bboxes[j][3])
        color = 'blue'
        if (j == imax[0]):
            color = 'red'

        file.write(
            '<svg class="two"><rect x="{0}" y="{1}" width="{2}" height="{3}" stroke="{4}" fill="none"/></svg>'.format(x, y, width, height, color))

    file.write('</div></td>\n')
    file.write('<td><p> Question: ' + questList[i] + '</p></td>\n')
    file.write('<td><p> Answer: ' + ansList[i] + '</p></td>\n')
    file.write('<td><p> Prediction: ' + str(output_max) + '</p></td>\n')
    file.write('\t</tr>\n')


    if ( np.fabs(ans-output_max) <  0.5):
        score += 1

    ans_stat.append((ans, output_max))

    if (ans > 0):
        num_yes_gt += 1

    if (output_max > 0.5):
        num_yes_pred += 1

    nQuestValid += 1

    sys.stdout.write("\r \r Evaluation:\t{0}/{1}\tAccumulated score: {2}".format( i, nQuest, score ) )
    sys.stdout.flush()
    time.sleep(0.01)

file.write('</table>\n</body>\n<html>')
file.close()

fileLog.write("Number of validated questions: {}\n".format(nQuestValid))
fileLog.write("Mean score is: {}\n".format(score/float(nQuestValid)))
fileLog.write("GT answers stat:\tyes: {0}%\tno: {1}%\n".format( 100*float(num_yes_gt)/float(nQuestValid),
                                                      100*(1-float(num_yes_gt)/float(nQuestValid))))
fileLog.write("Predicted answers stat:\tyes: {0}%\tno: {1}%\n".format( 100*float(num_yes_pred)/float(nQuestValid),
                                                             100*(1-float(num_yes_pred)/float(nQuestValid))))
fileLog.close()

print("\nEvaluation is done!")
print("Mean score is: {}".format(score/float(nQuestValid)))
print("GT answers stat:\tyes: {0}%\tno: {1}%".format( 100*float(num_yes_gt)/float(nQuestValid),
                                                      100*(1-float(num_yes_gt)/float(nQuestValid))))
print("Predicted answers stat:\tyes: {0}%\tno: {1}%".format( 100*float(num_yes_pred)/float(nQuestValid),
                                                             100*(1-float(num_yes_pred)/float(nQuestValid))))

