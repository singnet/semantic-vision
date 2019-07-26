#include "getBOW.h"

void getBowVoc(vector<Mat> baseDescs, Mat descs, Mat &vocabulary, int clusterCount, vector<Mat> * processedBase, Mat &processedBaseMat)
{
    BOWKMeansTrainer bowtrainer(clusterCount);
    Mat fdescs, oneFDesc;
    descs.convertTo(fdescs, CV_32F);
    bowtrainer.add(fdescs);
    vocabulary = bowtrainer.cluster();
    BFMatcher* matcher = new BFMatcher;
    BOWImgDescriptorExtractor bowide(matcher);
    bowide.setVocabulary(vocabulary);
    for (auto& oneDesc : baseDescs) {
        oneDesc.convertTo(oneFDesc, CV_32F);
        vector<vector<int>> idxs;
        Mat bowDesc;
        bowide.compute(oneFDesc, bowDesc, &idxs);
        processedBase->push_back(bowDesc);
        processedBaseMat.push_back(bowDesc);
    }
}

vector<vector<DMatch>> processOneDesc(Mat oneDesc, Mat vocabulary, Mat processedBase, int num)
{
    BFMatcher * matcher = new BFMatcher;
    BOWImgDescriptorExtractor bowide(matcher);
    bowide.setVocabulary(vocabulary);
    Mat bowDesc, oneFDesc;
    oneDesc.convertTo(oneFDesc, CV_32F);
    bowide.compute(oneFDesc, bowDesc);
    vector<vector<DMatch>> matches;
    matcher->knnMatch(bowDesc, processedBase,matches,num);
    return matches;
}