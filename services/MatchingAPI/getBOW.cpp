#include "getBOW.h"

void getBowVoc(vector<Mat> baseDescs, Vocabulary * vocabulary, vector<fBow> * processedBase)
{
    fbow::VocabularyCreator::Params params;
    params.k = 10;
    params.L = 6;
    params.nthreads=4;
    params.maxIters=3;
    params.verbose=false;
    srand(0);
    VocabularyCreator voc_creator;
    voc_creator.create((*vocabulary),baseDescs,"", params);
    for (auto& oneFeature : baseDescs)
        (*processedBase).push_back((*vocabulary).transform(oneFeature));
}

vector<pair<double, int>> processOneDesc(vector<fBow> processedBase, Mat oneDesc, Vocabulary &vocabulary, int num)
{
    fBow vv = vocabulary.transform(oneDesc);
    vector<pair<double, int>> scores;
    for (int j = 0; j < processedBase.size(); j++)
        scores.emplace_back(vv.score(vv, processedBase[j]), j);
    sort(scores.begin(), scores.end());
    vector<pair<double, int>> result;
    for (int i = 0; i < num; i++)
        result.push_back(scores[scores.size()-1-i]);
    return result;
}