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

pathVocabFile = '/mnt/fileserver/shared/datasets/at-on-at-data/yesno_predadj_words.txt'
pathFeaturesTrainParsed = '/mnt/fileserver/shared/datasets/at-on-at-data/train2014_parsed_features'
pathFeaturesValParsed = '/mnt/fileserver/shared/datasets/at-on-at-data/val2014_parsed_features'
pathDataTrainFile = '/mnt/fileserver/shared/datasets/at-on-at-data/train2014_questions_parsed.txt'
pathDataValFile = '/mnt/fileserver/shared/datasets/at-on-at-data/val2014_questions_parsed.txt'

pathPickledTrainFeatrues = '/mnt/fileserver/shared/datasets/at-on-at-data/COCO_train2014_yes_no.pkl'
pathPickledValFeatrues = '/mnt/fileserver/shared/datasets/at-on-at-data/COCO_val2014_yes_no.pkl'


pathSaveModel = './saved_models_01'
if os.path.isdir(pathSaveModel) is False:
    os.mkdir(pathSaveModel)

FILE_PREFIX_TRAIN = 'COCO_train2014_'
FILE_PREFIX_VAL = 'COCO_val2014_'
IMAGE_ID_FIELD_NAME = 'imageId'


# pathFeaturesTrainParsed = pathFeaturesValParsed
# pathDataTrainFile = '/home/mvp/Desktop/SingularityNET/datasets/VisualQA/balanced_real_images/val2014_questions_parsed.txt'
# FILE_PREFIX = 'COCO_val2014_'

input_size = 2048
nBBox = 36
featOffset = 10
nEpoch = 200
learning_rate = 1e-2
lr_decay_iter = 30
eps = 1e-16

isLoadPickledFeatures = True
isReduceSet = False

# Load vocabulary
vocab = []
with open(pathVocabFile, 'r') as filehandle:
    vocab = [current_place.rstrip() for current_place in filehandle.readlines()]

def getWords(groundedFormula):
    words = re.split(r', ', groundedFormula[groundedFormula.find("(") + 1:groundedFormula.find(")")])
    return words


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

nets = NetsVocab.fromWordsVocabulary(vocab, input_size, device)

# Continue training from saved checkpoint
# checkpoint = torch.load(pathSaveModel + '/model_bkp.pth.tar')
# mean_loss = checkpoint['mean_loss']
# ep = checkpoint['epoch']
# nets.NetsVocab.fromStateDict(device, checkpoint['state_dict'])



# Prepare training data

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
# Drop duplicates and sort
imgIdSet = sorted(set(imgIdList))


df_val = pd.read_csv(pathDataValFile, header=0, sep='\s*\::',  engine='python')
df_quest_val = df_val.loc[(df_val['questionType'] == 'yes/no') & (df_val['relexFormula'] == '_predadj(A, B)')]
df_quest_val = df_quest_val.sort_values(['imageId'], ascending=[True])
df_quest_val = df_quest_val.reset_index(drop=True)

imgIdList_val = df_quest_val[IMAGE_ID_FIELD_NAME].tolist()
# Drop duplicates and sort
imgIdSet_val = sorted(set(imgIdList_val))


if isLoadPickledFeatures is True:
    data_feat = vqp.load_pickled_features(pathPickledTrainFeatrues)
else:
    # !! FOR DEBUG LOAD ONLY 1% OF DATA !!! HARDCODED INSIDE vpq.load_parsed_features !!!!
    data_feat =  vqp.load_parsed_features(pathFeaturesTrainParsed, imgIdSet, filePrefix=FILE_PREFIX_TRAIN, reduce_set=isReduceSet)


# Get list with binary answers yes = 1, no = 0
nQuest = df_quest.shape[0]
ansList = df_quest['answer'].tolist()
num_yes_gt = 0
for i in range(nQuest):
    if ansList[i] == 'yes':
        ansList[i] = 1
        num_yes_gt += 1
    else:
        ansList[i] = 0
#########################################################################

# Prepare validation data

# Load bbox features
#df = pd.read_csv(pathDataTrainFile, header=0, sep='\s*\::',  engine='python')


if isLoadPickledFeatures is True:
    data_feat_val = vqp.load_pickled_features(pathPickledValFeatrues)
else:
    # !! FOR DEBUG LOAD ONLY 1% OF DATA !!! HARDCODED INSIDE vpq.load_parsed_features !!!!
    data_feat_val = vqp.load_parsed_features(pathFeaturesValParsed, imgIdSet_val, filePrefix=FILE_PREFIX_VAL,
                                             reduce_set=isReduceSet)

nQuest_val = df_quest_val.shape[0]
questList_val = df_quest_val['question'].tolist()
ansList_val = df_quest_val['answer'].tolist()
ansListBin_val = []
for i in range(nQuest_val):
    if ansList_val[i] == 'yes':
        ansListBin_val.append(1)
    else:
        ansListBin_val.append(0)

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!!
nQuest_val = len(data_feat_val)

##########################################################################



fileLog = open('log_train_01.txt', 'w')
fileLog.write("Number of training questions: {}\n".format(nQuest))
fileLog.write("Number of training images: {}\n".format(len(imgIdSet)))
fileLog.write("GT answers stat:\tyes: {0}%\tno: {1}%\n".format( 100*float(num_yes_gt)/float(nQuest),
                                                      100*(1-float(num_yes_gt)/float(nQuest))))

fileLogNumbers = open('log_train_01_nums.txt', 'w')
fileLogNumbers.write("# epoch\tmean_loss\ttrain_score\tval_score\n")

# !! FOR DEBUG LOAD ONLY 1% OF DATA !!!
nQuest = len(data_feat)
min_loss = 1e8
max_score_val = 0
lr = learning_rate
for e in range(nEpoch):
    mean_loss = 0.
    score = 0
    nQuestValid = 0

    for i in range(nQuest):
        words = getWords(df_quest.loc[i, 'groundedFormula'])

        # get img bbox features
        img_id = df_quest.loc[i, 'imageId']
        # ind = imgIdList.index(img_id)
        ind = imgIdSet.index(img_id)
        f_input = data_feat[ind][1][:, featOffset:]
        inputs = Variable(torch.Tensor(f_input)).to(device)


        # Feed each bbox feature in the batch (36) to selected nets and multiply output probabilities
        output = nets.feed_forward(nBBox, inputs, words)

        ans = torch.from_numpy(np.asarray(ansList[i], dtype=np.float32)).to(device)

        sum = torch.sum(output) + torch.Tensor([eps]).to(device)
        #s = torch.div(output, sum)
        sum_sq = torch.sum(torch.mul(output, output))
        output = sum_sq / sum

        loss = F.binary_cross_entropy(output, ans)

        params = nets.getParams(words)

        # adjust learning rate
        if e % lr_decay_iter == 0:
            lr = learning_rate * (1 - float(e) / float(nEpoch)) ** 0.9

        optimizer = torch.optim.SGD(params, lr=lr, momentum=0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        res_loss = loss.data.cpu().numpy()
        mean_loss += res_loss

        res_predict = np.max( output.data.cpu().numpy() )
        abs_diff = np.fabs(ansList[i] - res_predict)
        if ( abs_diff < 0.5):
            score += 1

        sys.stdout.write("\r \r Training:\tepoch: {0}/{1}\tquestion: {2}/{3}\tloss: {4}".format( e, nEpoch, i, nQuest, res_loss) )
        sys.stdout.flush()
        time.sleep(0.01)

        nQuestValid += 1

    mean_loss /= float(nQuestValid)
    score = score/float(nQuestValid)
    print("\nEpoch: {0}/{1}\t lr: {2}\tMean loss: {3}\tScore: {4}%\n".format( e, nEpoch, lr, mean_loss, 100*score))
    fileLog.write("Epoch: {0}/{1}\tlr: {2}\tMean loss: {3}\tScore: {4}%\n".format(e, nEpoch, lr, mean_loss, 100 * score))

    # Validation
    if (e % 10) == 0:
        score_val = 0.
        nQuestValid_val = 0

        for i in range(nQuest_val):
            words_val = getWords(df_quest_val.loc[i, 'groundedFormula'])

            # get img bbox features
            img_id = df_quest_val.loc[i, 'imageId']

            ind_val = imgIdSet_val.index(img_id)
            bboxes = data_feat_val[ind_val][1][:, 0:4]
            f_input_val = data_feat_val[ind_val][1][:, featOffset:]
            inputs_val = Variable(torch.Tensor(f_input_val)).to(device)

            # Feed each bbox feature in the batch (36) to selected nets and multiply output probabilities
            output_val = nets.feed_forward(nBBox, inputs_val, words_val)

            sum_val = torch.sum(output_val) + torch.Tensor([eps]).to(device)
            #s_val = torch.div(output_val, sum_val)
            sum_sq_val = torch.sum(torch.mul(output_val, output_val))
            output_val = sum_sq_val / sum_val

            output_max_val = output_val.data.cpu().numpy()
            abs_diff = math.fabs(ansListBin_val[i] - output_max_val)
            if (abs_diff < 0.5):
                score_val += 1

            nQuestValid_val += 1

            sys.stdout.write("\r \r Evaluation:\t{0}/{1}\t\tAccumulated score: {2}".format(i, nQuest_val, score_val))
            sys.stdout.flush()
            time.sleep(0.01)

        score_val = score_val / float(nQuestValid_val)

        print("\nEvaluation is done!")
        print("Mean score is: {}%\n".format(100*score_val))

        fileLog.write(
            "\n Evaluation at {0}/{1} epoch gives score:\t{2}%\n\n".format(e, nEpoch, 100 * score_val))

        fileLogNumbers.write("{0}\t{1}\t{2}\t{3}\n".format(e, mean_loss, 100*score, 100*score_val))

        if (max_score_val < score_val):
            state = {'epoch': e, 'state_dict': nets.state_dict(), 'optimizer': optimizer.state_dict(),
                     'mean_loss': mean_loss}
            filename = pathSaveModel + '/model_01_max_score_val.pth.tar'
            torch.save(state, filename)

            fileLog.write(
                "\n Saving model at {0}/{1} epoch with val_score: {2}%\n\n".format(e, nEpoch, 100 * score_val))

            max_score_val = score_val

    # if (mean_loss < min_loss):
    #     state = {'epoch': e, 'state_dict': nets.state_dict(), 'optimizer' : optimizer.state_dict(), 'mean_loss' : mean_loss }
    #     filename = pathSaveModel + '/model_01.pth.tar'
    #     torch.save(state, filename)
    #
    #     fileLog.write("\n Saving model at {0}/{1} epoch with mean loss: {2}\tscore: {3}%\n\n".format(e, nEpoch, mean_loss, 100 * score))
    #     print(
    #         "\n Saving model at {0}/{1} epoch with mean loss: {2}\tscore: {3}%\n\n".format(e, nEpoch, mean_loss,
    #                                                                                        100 * score))

fileLogNumbers.close()
fileLog.close()
print("Training is done!")

