#ifndef MATCHINGAPI_TRANSFORMATIONS_H
#define MATCHINGAPI_TRANSFORMATIONS_H

#include "Includes.h"
#include "CheckParam.h"
#include "additionalTransformations.h"

class ITransform
{
public:
    virtual void setParameters(map<string, double> params) = 0;
    virtual vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) = 0;
};

ITransform* ChooseTransform(const char *name);

class AffineTransform : public ITransform
{
public:
    AffineTransform();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static AffineTransform* create();
};

class PerspectiveTransform : public ITransform
{
public:
    PerspectiveTransform();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static PerspectiveTransform* create();
};

class EssentialMatrix : public ITransform
{
public:
    EssentialMatrix();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static EssentialMatrix* create();
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
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static FundamentalMatrix *create();

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
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static Homography *create();

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
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static Similarity *create();

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
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static Shift *create();
};

class ShiftScale : public ITransform {
public:
    ShiftScale();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static ShiftScale *create();
};

class ShiftRot : public ITransform {
public:
    ShiftRot();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static ShiftRot *create();
};

class Poly : public ITransform
{
public:
    Poly();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static Poly *create();
};

class Bilinear : public ITransform
{
public:
    Bilinear();
    void setParameters(map<string, double> params) override;
    vector<double> getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches) override;
    static Bilinear *create();
};

#endif //MATCHINGAPI_TRANSFORMATIONS_H
