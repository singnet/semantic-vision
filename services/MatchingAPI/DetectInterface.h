#ifndef MATCHINGAPI_DETECTINTERFACE_H
#define MATCHINGAPI_DETECTINTERFACE_H

#include "Includes.h"
#include "CheckParam.h"

class IDetector
{
public:
    virtual void setParameters(map<string, double> params) = 0;
    virtual vector<KeyPoint> getPoints(Mat image) = 0;
};

IDetector* ChooseDetector(const char *name);

class ORBDetector: public IDetector
{
public:
    ORBDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(Mat image) override;
    static ORBDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static KAZEDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static AKAZEDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static AGASTDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static GFTDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static MSERDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static BRISKDetector* create();

private:
    int  	thresh;
    int  	octaves;
};

class FASTDetector: public IDetector
{
public:
    FASTDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(Mat image) override;
    static FASTDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static BLOBDetector* create();
};

class STARDetector: public IDetector
{
public:
    STARDetector();
    void setParameters(map<string, double> params) override;
    vector<KeyPoint> getPoints(Mat image) override;
    static STARDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static MSDSDetector* create();

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
    vector<KeyPoint> getPoints(Mat image) override;
    static HLFDDetector* create();

private:
    int  	numOctaves;
    float  	corn_thresh;
    float  	DOG_thresh;
    int  	maxCorners;
    int  	num_layers;
};

#endif