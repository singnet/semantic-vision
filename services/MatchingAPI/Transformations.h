#ifndef MATCHINGAPI_TRANSFORMATIONS_H
#define MATCHINGAPI_TRANSFORMATIONS_H

#include "Includes.h"
#include "CheckParam.h"
#include "additionalTransformations.h"

class ITransform
{
public:
    virtual void setParameters(map<string, double> params) = 0;
    virtual vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) = 0;
    virtual void releaseTransform() = 0;
};

ITransform* ChooseTransform(const char *name);

class AffineTransform : public ITransform
{
public:
    AffineTransform();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static AffineTransform* create();
    void releaseTransform() override;
};

class PerspectiveTransform : public ITransform
{
public:
    PerspectiveTransform();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static PerspectiveTransform* create();
    void releaseTransform() override;
};

class EssentialMatrix : public ITransform
{
public:
    EssentialMatrix();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static EssentialMatrix* create();
    void releaseTransform() override;
private:
    char *methodNames[9] = {};
    double focal;
    double pp_x;
    double pp_y;
    int method;
    double prob;
    double threshold;
};

class FundamentalMatrix : public ITransform {
public:
    FundamentalMatrix();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static FundamentalMatrix *create();
    void releaseTransform() override;

private:
    char *methodNames[9] = {};
    int method;
    double ransacReprojThreshold;
    double confidence;
};

class Homography : public ITransform {
public:
    Homography();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static Homography *create();
    void releaseTransform() override;

private:
    char *methodNames[17] = {};
    int method;
    double ransacReprojThreshold;
    int maxIters;
    double confidence;
};

class Similarity : public ITransform {
public:
    Similarity();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static Similarity *create();
    void releaseTransform() override;

private:
    char *methodNames[9] = {};
    int method;
    double ransacReprojThreshold;
    int maxIters;
    double confidence;
    int refineIters;
};

class Shift : public ITransform {
public:
    Shift();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static Shift *create();
    void releaseTransform() override;
};

class ShiftScale : public ITransform {
public:
    ShiftScale();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static ShiftScale *create();
    void releaseTransform() override;
};

class ShiftRot : public ITransform {
public:
    ShiftRot();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static ShiftRot *create();
    void releaseTransform() override;
};

class Poly : public ITransform
{
public:
    Poly();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static Poly *create();
    void releaseTransform() override;
};

class Bilinear : public ITransform
{
public:
    Bilinear();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches, Mat src, Mat& image) override;
    static Bilinear *create();
    void releaseTransform() override;
};

#endif //MATCHINGAPI_TRANSFORMATIONS_H
