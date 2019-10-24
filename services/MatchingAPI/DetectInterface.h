#ifndef MATCHINGAPI_DETECTINTERFACE_H
#define MATCHINGAPI_DETECTINTERFACE_H

#include "Utilities.h"
#include "CheckParam.h"

class IDetector
{
public:
    virtual void setParameters(map<string, double> params) = 0;
    virtual vector<KeyPoint> getPoints(string image) = 0;
    virtual void releaseDetector() = 0;
};

IDetector* ChooseDetector(const char *name);

class ORBDetector: public IDetector
{
public:
    ORBDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static ORBDetector* create();
    void releaseDetector() override;

private:
    int nfeatures;
    float scaleFactor;
    int nlevels;
    int edgeThreshold;
    int firstLevel;
};

class KAZEDetector: public IDetector
{
public:
    KAZEDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static KAZEDetector* create();
    void releaseDetector() override;

private:
    float  	threshold;
    int  	nOctaves;
    int  	nOctaveLayers;
    int  	diffusivity;
    char *diffTypes[4] = {};
};

class AKAZEDetector: public IDetector
{
public:
    AKAZEDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static AKAZEDetector* create();
    void releaseDetector() override;

private:
    float threshold;
    int  	nOctaves;
    int  	nOctaveLayers;
    int  	diffusivity;
    char *diffTypes[4] = {};
};

class AGASTDetector: public IDetector
{
public:
    AGASTDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static AGASTDetector* create();
    void releaseDetector() override;

private:
    int 	threshold;
    bool  	nonmaxSuppression;
    int    type;
    char *types[4] = {};
};

class GFTDetector: public IDetector
{
public:
    GFTDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static GFTDetector* create();
    void releaseDetector() override;

private:
    int  	maxCorners;
    double  	qualityLevel;
    double  	minDistance;
    int  	blockSize;
    bool  	useHarrisDetector;
    double  	k;
};

class MSERDetector: public IDetector
{
public:
    MSERDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static MSERDetector* create();
    void releaseDetector() override;

private:
    int  	_delta;
    int  	_min_area;
    int  	_max_area;
    double  	_max_variation;
    double  	_min_diversity;
    int  	_max_evolution;
    double  	_area_threshold;
    double  	_min_margin;
    int  	_edge_blur_size;
};

class BRISKDetector: public IDetector
{
public:
    BRISKDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static BRISKDetector* create();
    void releaseDetector() override;

private:
    int  	thresh;
    int  	octaves;
};

class FASTDetector: public IDetector
{
public:
    FASTDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static FASTDetector* create();
    void releaseDetector() override;

private:
    int  	threshold;
    bool  	nonmaxSuppression;
    int  	type;
    char*   types[3] = {};
};

class BLOBDetector: public IDetector
{
public:
    BLOBDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static BLOBDetector* create();
    void releaseDetector() override;
};

class STARDetector: public IDetector
{
public:
    STARDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static STARDetector* create();
    void releaseDetector() override;

private:
    int  	maxSize;
    int  	responseThreshold;
    int  	lineThresholdProjected;
    int  	lineThresholdBinarized;
    int  	suppressNonmaxSize;
};

class MSDSDetector: public IDetector
{
public:
    MSDSDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static MSDSDetector* create();
    void releaseDetector() override;

private:
    int  	m_patch_radius;
    int  	m_search_area_radius;
    int  	m_nms_radius;
    int  	m_nms_scale_radius;
    float  	m_th_saliency;
    int  	m_kNN;
    float  	m_scale_factor;
    int  	m_n_scales;
    bool  	m_compute_orientation;
};

class HLFDDetector: public IDetector
{
public:
    HLFDDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static HLFDDetector* create();
    void releaseDetector() override;

private:
    int  	numOctaves;
    float  	corn_thresh;
    float  	DOG_thresh;
    int  	maxCorners;
    int  	num_layers;
};

class MPDetector: public IDetector
{
public:
    MPDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static MPDetector* create();
    void releaseDetector() override;

private:
    float threshold;
};

class SPDetector: public IDetector
{
public:
    SPDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(string image) override;
    static SPDetector* create();
    void releaseDetector() override;

private:
    float threshold;
};

#endif