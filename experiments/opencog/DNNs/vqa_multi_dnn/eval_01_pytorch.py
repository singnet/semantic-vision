import pandas as pd
import vqa_data_parser as vqp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import os, sys, time, re
import math
from netsvocabulary import NetsVocab


# FOR RUNNING ON K4
# pathVocabFile = '/home/shared/datasets/yesno_predadj_words.txt'
# pathFeaturesParsed = '/home/shared/datasets/VisualQA/Attention-on-Attention-data/val2014_parsed_features'
# pathQuestFile = '/home/shared/datasets/val2014_questions_parsed.txt'
# pathImgs = '/home/shared/datasets/val2014'


#
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


pathSaveModel = './saved_models_01/99.16_67.65'

isReduceSet = False

input_size = 2048
nBBox = 36
featOffset = 10


# Load vocabulary
vocab = []
with open(pathVocabFile, 'r') as filehandle:
    vocab = [current_place.rstrip() for current_place in filehandle.readlines()]

Nnets = len(vocab)

def getWords(groundedFormula):
    words = re.split(r', ', groundedFormula[groundedFormula.find("(") + 1:groundedFormula.find(")")])
    return words


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print('Loading model...')
nets = NetsVocab(vocab, input_size, device)
checkpoint = torch.load(pathSaveModel + '/model_01_max_score_val.pth.tar')
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
imgIdSet = sorted(set(imgIdList))

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!! HARDCODED INSIDE vpq.load_parsed_features !!!!
data_feat =  vqp.load_parsed_features(pathFeaturesParsed, imgIdSet, filePrefix=FILE_PREFIX, reduce_set=isReduceSet)

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

fileLog = open("log_val_01.txt", 'w')

file = open("eval_01_results.html", 'w')
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
    words = getWords(df_quest.loc[i, 'groundedFormula'])

    # get img bbox features
    img_id = df_quest.loc[i, 'imageId']
    filePath = pathImgs + '/' + FILE_PREFIX
    nZeros = int((id_len - 1) - math.floor(math.log10(img_id)))
    for _ in range(0, nZeros):
        filePath = filePath + '0'

    filePath = filePath + str(img_id) + '.jpg'


    ind = imgIdSet.index(img_id)
    bboxes = data_feat[ind][1][:, 0:4]
    f_input = data_feat[ind][1][:, featOffset:]
    inputs = Variable(torch.Tensor(f_input)).to(device)


    # Feed each bbox feature in the batch (36) to selected nets and multiply output probabilities
    output = nets.feed_forward(nBBox, inputs, words)

    ans = np.asarray(ansListBin[i], dtype=np.float32)
    _, idx_max = torch.max(output, 0)
    imax = idx_max.data.cpu().numpy()

    sum = torch.sum(output)
    s = torch.div(output, sum)
    sum_sq = torch.sum(torch.mul(output, output))
    output = sum_sq / sum

    output_max = output.data.cpu().numpy()

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

    abs_diff = math.fabs(ansListBin[i]-output_max)
    if ( abs_diff <  0.5):
        score += 1

    ans_stat.append((ans, output_max))

    if (ans > 0):
        num_yes_gt += 1

    if (output_max > 0.5):
        num_yes_pred += 1

    nQuestValid += 1

    sys.stdout.write("\r \r Evaluation:\t{0}/{1}\t\tAccumulated score: {2}".format( i, nQuest, score ) )
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

