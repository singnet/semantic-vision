#ifndef MATCHINGAPI_FEATURESINTERFACE_H
#define MATCHINGAPI_FEATURESINTERFACE_H

#include "CheckParam.h"

class IDescribe
{
public:
    virtual void setParameters(map<string, double> parameters) = 0;
    virtual Mat getFeatures(vector<KeyPoint> input, Mat image) = 0;
    ~IDescribe() = default;
};

IDescribe* ChooseFeatures(const char *name);

class orbFeatures: public IDescribe
{
public:
    orbFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static orbFeatures* create();
    ~orbFeatures()=default;
private:
    int WTA_K;
};

class kazeFeatures: public IDescribe
{
public:
    kazeFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static kazeFeatures* create();
    ~kazeFeatures()=default;
private:
    int extended;
    int upright;
    int diffusivityType;
    char *diffTypes[4] = {};
};

class akazeFeatures: public IDescribe
{
public:
    akazeFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static akazeFeatures* create();
    ~akazeFeatures()=default;
private:
    int descriptor_type;
    int descriptor_size;
    int diffusivityType;
    int descriptor_channels;
    char *diffTypes[4] = {};
    char *discTypes[4] = {};
};

class briskFeatures: public IDescribe
{
public:
    briskFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static briskFeatures* create();
    ~briskFeatures()=default;
private:
    float patternScale;
};

class briefFeatures: public IDescribe
{
public:
    briefFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static briefFeatures* create();
    ~briefFeatures()=default;
private:
    int bytes;
    int use_orientation;
};

class freakFeatures: public IDescribe
{
public:
    freakFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static freakFeatures* create();
    ~freakFeatures()=default;
private:
    int orientationNormalized;
    int scaleNormalized;
    float patternScale;
    int nOctaves;
};

class lucidFeatures: public IDescribe
{
public:
    lucidFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static lucidFeatures* create();
    ~lucidFeatures()=default;
private:
    int lucid_kernel;
    int blur_kernel;
};


class latchFeatures: public IDescribe
{
public:
    latchFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static latchFeatures* create();
    ~latchFeatures()=default;
private:
    int bytes;
    int rotationInvariance;
    int half_ssd_size;
    double sigma ;
};

class daisyFeatures: public IDescribe
{
public:
    daisyFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static daisyFeatures* create();
    ~daisyFeatures()=default;
private:
    float radius;
    int q_radius;
    int q_theta;
    int q_hist;
    int norm;
    InputArray H = noArray();
    int interpolation;
    int use_orientation;
    char* descTypes[4] = {};
};

class vggFeatures: public IDescribe
{
public:
    vggFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static vggFeatures* create();
    ~vggFeatures()=default;
private:
    int desc;
    float isigma;
    int img_normalize;
    int use_scale_orientation;
    float scale_factor;
    int dsc_normalize;
    char* descTypes[4] = {};
};

class boostFeatures: public IDescribe
{
public:
    boostFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static boostFeatures* create();
    ~boostFeatures()=default;

private:
    int desc;
    int use_scale_orientation;
    float scale_factor;
    char *descTypes[303] = {};
};

class pctFeatures: public IDescribe
{
public:
    pctFeatures();
    void setParameters(map<string, double> parameters) override;
    Mat getFeatures(vector<KeyPoint> input, Mat image) override;
    static pctFeatures* create();
    ~pctFeatures()=default;
private:
    int initSampleCount;
    int initSeedCount;
    int pointDistribution;
    char* distTypes[3] = {};
};
#endif