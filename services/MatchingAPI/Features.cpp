#define NPY_NO_DEPRECATED_APINPY_1_7_API_VERSION

#include "FeaturesInterface.h"
#include <limits>

#include <Python.h>
#include "numpy/arrayobject.h"

#include <memory>

int boolRange[] = {0, 1};

int imax_value = std::numeric_limits<int>::max();
float fmax_value = std::numeric_limits<float>::max();
double dmax_value = std::numeric_limits<double>::max();

static Mat getMat(string imageBytes)
{
    size_t length = imageBytes.size();
    Mat imageMat;
    vector<char> data((char *) imageBytes.c_str(), (char *) imageBytes.c_str() + length);
    imageMat = imdecode(data, IMREAD_UNCHANGED);
    return imageMat;
}

IDescribe* ChooseFeatures(const char *name)
{
    if (strncmp(name, (char*)"ORB", 3) == 0)
    {
        cout << "Using ORB features" << endl;
        return orbFeatures::create();
    }
    else if (strncmp(name, (char*)"AKAZE", 5) == 0)
    {
        cout << "Using AKAZE features" << endl;
        return akazeFeatures::create();
    }
    else if (strncmp(name, (char*)"KAZE", 4) == 0)
    {
        cout << "Using KAZE features" << endl;
        return kazeFeatures::create();
    }
    else if (strncmp(name, (char*)"BRISK", 4) == 0)
    {
        cout << "Using BRISK features" << endl;
        return briskFeatures::create();
    }
    else if (strncmp(name, (char*)"BRIEF", 4) == 0)
    {
        cout << "Using BRIEF features" << endl;
        return briefFeatures::create();
    }
    else if (strncmp(name, (char*)"FREAK", 4) == 0)
    {
        cout << "Using FREAK features" << endl;
        return freakFeatures::create();
    }
    else if (strncmp(name, (char*)"LUCID", 4) == 0)
    {
        cout << "Using LUCID features" << endl;
        return lucidFeatures::create();
    }
    else if (strncmp(name, (char*)"LATCH", 4) == 0)
    {
        cout << "Using LATCH features" << endl;
        return latchFeatures::create();
    }
    else if (strncmp(name, (char*)"DAISY", 4) == 0)
    {
        cout << "Using DAISY features" << endl;
        return daisyFeatures::create();
    }
    else if (strncmp(name, (char*)"VGG", 4) == 0)
    {
        cout << "Using VGG features" << endl;
        return vggFeatures::create();
    }
    else if (strncmp(name, (char*)"BOOST", 4) == 0)
    {
        cout << "Using BOOST features" << endl;
        return boostFeatures::create();
    }
    else if (strncmp(name, (char*)"PCT", 4) == 0)
    {
        cout << "Using PCT features" << endl;
        return pctFeatures::create();
    }
    else if (strncmp(name, (char*)"Superpoint", 4) == 0)
    {
        cout << "Using Superpoint features" << endl;
        return superpointFeatures::create();
    }
    else
    {
        cout << "No descriptor was chosen or there is a mistake in the name. Using ORB by default" << endl;
        return orbFeatures::create();
    }
}

orbFeatures::orbFeatures()
{
    WTA_K = 2;
}

void orbFeatures::setParameters(map<string, double> parameters)
{
    int range[] = {2, 3, 4};
    CheckParam <int> (parameters, (char*)"WTA_K", &WTA_K, range, 3);
}

orbFeatures* orbFeatures::create()
{
    auto* ptr_Orb = new orbFeatures();
    return ptr_Orb;
}

Mat orbFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<ORB> featureDescriptor = ORB::create(500, 1.2f, 8, 31, 0, WTA_K);
    featureDescriptor->compute(imageMat, (*input), descriptors);

    return descriptors;
}

void orbFeatures::releaseDescriptor() {delete this;}

kazeFeatures::kazeFeatures()
{
    upright = 0;
    extended = 0;
    diffusivityType = 1;
    diffTypes[0] = (char*)"DIFF_PM_G1";
    diffTypes[1] = (char*)"DIFF_PM_G2";
    diffTypes[2] = (char*)"DIFF_WEICKERT";
    diffTypes[3] = (char*)"DIFF_CHARBONNIER";
}

void kazeFeatures::setParameters(map<string, double> parameters)
{

    CheckParam <int> (parameters, (char*)"upright", &upright, boolRange, 2);
    CheckParam <int> (parameters, (char*)"extended", &extended, boolRange, 2);
    int diffRange[] = {0, 1, 2, 3};
    CheckParam <int> (parameters, (char*)"diffusivityType", &diffusivityType, diffRange, 4);
    cout << " - " << diffTypes[diffusivityType] << endl;
}

kazeFeatures* kazeFeatures::create()
{
    auto* ptr_Kaze = new kazeFeatures();
    return ptr_Kaze;
}

Mat kazeFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<KAZE> featureDescriptor = KAZE::create(extended, upright, 0.001f, 4, 4, (KAZE::DiffusivityType)diffusivityType);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}
void kazeFeatures::releaseDescriptor() {delete this;}


akazeFeatures::akazeFeatures()
{
    descriptor_type = 5;
    descriptor_size = 0;
    diffusivityType = 1;
    descriptor_channels = 3;

    diffTypes[0] = (char*)"DIFF_PM_G1";
    diffTypes[1] = (char*)"DIFF_PM_G2";
    diffTypes[2] = (char*)"DIFF_WEICKERT";
    diffTypes[3] = (char*)"DIFF_CHARBONNIER";

    discTypes[0] = (char*)"DESCRIPTOR_KAZE_UPRIGHT";
    discTypes[1] = (char*)"DESCRIPTOR_KAZE";
    discTypes[2] = (char*)"DESCRIPTOR_MLDB_UPRIGHT";
    discTypes[3] = (char*)"DESCRIPTOR_MLDB";
}

void akazeFeatures::setParameters(map<string, double> parameters)
{
    int descRange[] = {2, 3, 4, 5};
    CheckParam <int> (parameters, (char*)"descriptor_type", &descriptor_type, descRange, 4);
    cout << " - " << discTypes[descriptor_type-2];

    int channelRange[] = {1, 2, 3};
    CheckParam<int> (parameters, (char*)"descriptor_channels", &descriptor_channels, channelRange, 3);
    if((descriptor_type == 4 || descriptor_type == 5) && descriptor_channels != 3)
    {
        cout << "MLDB descriptor supports descriptor_channels = 3 only." << endl;
        descriptor_channels = 3;
    }

    CheckParam_no_ifin<int> (parameters, (char*)"descriptor_size", &descriptor_size, 0, 162*descriptor_channels);

    int diffRange[] = {0, 1, 2, 3};
    CheckParam<int> (parameters, (char*)"diffusivityType", &diffusivityType, diffRange, 4);
    cout << " - " << diffTypes[diffusivityType] << endl;
}

akazeFeatures* akazeFeatures::create()
{
    auto* ptr_Akaze = new akazeFeatures();
    return ptr_Akaze;
}

Mat akazeFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<AKAZE> featureDescriptor = AKAZE::create((AKAZE::DescriptorType)descriptor_type, descriptor_size,
            descriptor_channels, 0.001f, 4, 4, (KAZE::DiffusivityType)diffusivityType);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void akazeFeatures::releaseDescriptor() {delete this;}

briskFeatures::briskFeatures()
{
    patternScale=1.0f;
}

void briskFeatures::setParameters(map<string, double> parameters)
{
    CheckParam_no_ifin <float>(parameters, (char*)"patternScale", &patternScale, 1.f, fmax_value);
}

briskFeatures* briskFeatures::create()
{
    auto* ptr_Brisk = new briskFeatures();
    return ptr_Brisk;
}

Mat briskFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<BRISK> featureDescriptor = BRISK::create(30, 3, patternScale);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void briskFeatures::releaseDescriptor() {delete this;}

briefFeatures::briefFeatures()
{
    bytes = 32;
    use_orientation = 0;
}

void briefFeatures::setParameters(map<string, double> parameters)
{
    int byteRange[] = {16, 32, 64};
    CheckParam <int> (parameters, (char*)"bytes", &bytes, byteRange, 3);

    CheckParam <int> (parameters, (char*)"use_orientation", &use_orientation, boolRange, 2);
}

briefFeatures* briefFeatures::create()
{
    auto* ptr_Brief = new briefFeatures();
    return ptr_Brief;
}

Mat briefFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<BriefDescriptorExtractor> featureDescriptor = BriefDescriptorExtractor::create(bytes, use_orientation);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void briefFeatures::releaseDescriptor() {delete this;}

freakFeatures::freakFeatures()
{
    orientationNormalized = 1;
    scaleNormalized = 1;
    patternScale = 22.0f;
    nOctaves = 4;
}

void freakFeatures::setParameters(map<string, double> parameters)
{

    CheckParam <int> (parameters, (char*)"orientationNormalized", &orientationNormalized, boolRange, 2);

    CheckParam <int> (parameters, (char*)"scaleNormalized", &scaleNormalized, boolRange, 2);

    CheckParam_no_ifin <float> (parameters, (char*)"patternScale", &patternScale, 0, fmax_value);

    CheckParam_no_ifin <int> (parameters, (char*)"nOctaves", &nOctaves, 0, imax_value);
}

freakFeatures* freakFeatures::create()
{
    auto* ptr_Freak = new freakFeatures();
    return ptr_Freak;
}

Mat freakFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<FREAK> featureDescriptor = FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void freakFeatures::releaseDescriptor() {delete this;}

lucidFeatures::lucidFeatures()
{
    lucid_kernel = 1;
    blur_kernel = 2;
}

void lucidFeatures::setParameters(map<string, double> parameters)
{
    CheckParam_no_ifin <int> (parameters, (char*)"lucid_kernel", &lucid_kernel, 0, imax_value);

    CheckParam_no_ifin <int> (parameters, (char*)"blur_kernel", &blur_kernel, 0, imax_value);

}

lucidFeatures* lucidFeatures::create()
{
    auto* ptr_Lucid = new lucidFeatures();
    return ptr_Lucid;
}

Mat lucidFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<LUCID> featureDescriptor = LUCID::create(lucid_kernel, blur_kernel);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void lucidFeatures::releaseDescriptor() {delete this;}

latchFeatures::latchFeatures()
{
    bytes = 32;
    rotationInvariance = 1;
    half_ssd_size = 3;
    sigma = 2.0;
}

Mat latchFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<LATCH> featureDescriptor = LATCH::create(bytes, rotationInvariance, half_ssd_size, sigma);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

latchFeatures* latchFeatures::create()
{
    auto* ptr_Latch = new latchFeatures();
    return ptr_Latch;
}

void latchFeatures::setParameters(map<string, double> parameters)
{
    int byteRange[] = {64, 32, 16, 8, 4, 2, 1};
    CheckParam <int> (parameters, (char*)"bytes", &bytes, byteRange, 7);
    CheckParam <int> (parameters, (char*)"rotationInvariance", &rotationInvariance, boolRange, 2);

    CheckParam_no_ifin <int> (parameters, (char*)"half_ssd_size", &half_ssd_size, 0, imax_value);

    CheckParam_no_ifin <double> (parameters, (char*)"sigma", &sigma, 0, dmax_value);
}

void latchFeatures::releaseDescriptor() {delete this;}

daisyFeatures::daisyFeatures()
{
    radius = 15;
    q_radius = 3;
    q_theta = 8;
    q_hist = 8;
    norm = 100;
    interpolation = 1;
    use_orientation = 0;
    descTypes[0] = (char*)"DAISY::NRM_NONE";
    descTypes[1] = (char*)"DAISY::NRM_PARTIAL";
    descTypes[2] = (char*)"DAISY::NRM_FULL";
    descTypes[3] = (char*)"DAISY::NRM_SIFT";
}

void daisyFeatures::setParameters(map<string, double> parameters)
{

    CheckParam_no_ifin <float> (parameters, (char*)"radius", &radius, 0, fmax_value);

    CheckParam_no_ifin <int> (parameters, (char*)"q_radius", &q_radius, 1, 656);

    CheckParam_no_ifin <int> (parameters, (char*)"q_theta", &q_theta, 0, imax_value);

    CheckParam_no_ifin <int> (parameters, (char*)"q_hist", &q_hist, 1, imax_value);

    int descRange[] = {100, 101, 102, 103};
    CheckParam <int> (parameters, (char*)"norm", &norm, descRange, 4);
    cout << " - " << descTypes[norm-100];

    CheckParam <int> (parameters, (char*)"interpolation", &interpolation, boolRange, 2);

    CheckParam <int> (parameters, (char*)"use_orientation", &use_orientation, boolRange, 2);
}

daisyFeatures* daisyFeatures::create()
{
    auto* ptr_Daisy = new daisyFeatures();
    return ptr_Daisy;
}

Mat daisyFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<DAISY> featureDescriptor = DAISY::create(radius, q_radius, q_theta, q_hist,
            (DAISY::NormalizationType)norm, H, interpolation, use_orientation);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void daisyFeatures::releaseDescriptor() {delete this;}

vggFeatures::vggFeatures()
{
    desc = 100;
    isigma = 1.4f;
    img_normalize = 1;
    use_scale_orientation = 1;
    scale_factor = 6.25f;
    dsc_normalize = 0;
    descTypes[0] = (char*)"VGG_120";
    descTypes[1] = (char*)"VGG_80";
    descTypes[2] = (char*)"VGG_64";
    descTypes[3] = (char*)"VGG_48";
}

void vggFeatures::setParameters(map<string, double> parameters)
{
    int descRange[] = {100, 101, 102, 103};
    CheckParam <int> (parameters, (char*)"desc", &desc, descRange, 4);
    cout << " - " << descTypes[desc-100];

    CheckParam_no_ifin <float> (parameters, (char*)"isigma", &isigma, 1.f, 1000.f);

    CheckParam <int> (parameters, (char*)"img_normalize", &img_normalize, boolRange, 2);

    CheckParam <int> (parameters, (char*)"use_scale_orientation", &use_scale_orientation, boolRange, 2);

    CheckParam <int> (parameters, (char*)"dsc_normalize", &dsc_normalize, boolRange, 2);

    CheckParam_no_ifin <float> (parameters, (char*)"scale_factor", &scale_factor, 0.f, fmax_value);
}

vggFeatures* vggFeatures::create()
{
    auto* ptr_Vgg = new vggFeatures();
    return ptr_Vgg;
}

Mat vggFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<VGG> featureDescriptor = VGG::create(desc, isigma, img_normalize, use_scale_orientation, scale_factor,
            dsc_normalize);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void vggFeatures::releaseDescriptor() {delete this;}

boostFeatures::boostFeatures()
{
    desc = 302;
    use_scale_orientation = 1;
    scale_factor = 6.25f;
    descTypes[100] = (char*)"BGM";
    descTypes[101] = (char*)"BGM_HARD";
    descTypes[102] = (char*)"BGM_BILINEAR";
    descTypes[200] = (char*)"LBGM";
    descTypes[300] = (char*)"BINBOOST_64";
    descTypes[301] = (char*)"BINBOOST_128";
    descTypes[302] = (char*)"BINBOOST_256";
}

void boostFeatures::setParameters(map<string, double> parameters)
{
    int descRange[] = {100, 101, 102, 200, 300, 301, 302};
    CheckParam <int> (parameters, (char*)"desc", &desc, descRange, 4);
    cout << " - " << descTypes[desc];

    CheckParam <int> (parameters, (char*)"use_scale_orientation", &use_scale_orientation, boolRange, 2);

    CheckParam_no_ifin <float> (parameters, (char*)"scale_factor", &scale_factor, 0.f, fmax_value);
}

boostFeatures* boostFeatures::create()
{
    auto* ptr_Boost = new boostFeatures();
    return ptr_Boost;
}

Mat boostFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<BoostDesc> featureDescriptor = BoostDesc::create(desc, use_scale_orientation, scale_factor);
    featureDescriptor->compute(imageMat, (*input), descriptors);
    return descriptors;
}

void boostFeatures::releaseDescriptor() {delete this;}

pctFeatures::pctFeatures()
{
    initSampleCount = 2000;
    initSeedCount = 400;
    pointDistribution = 0;
    distTypes[0] = (char*)"UNIFORM";
    distTypes[1] = (char*)"REGULAR";
    distTypes[2] = (char*)"NORMAL";
}

void pctFeatures::setParameters(map<string, double> parameters)
{
    int distRange[] = {0, 1, 2};
    CheckParam <int> (parameters, (char*)"pointDistribution", &pointDistribution, distRange, 4);
    cout << " - " << distTypes[pointDistribution];

    CheckParam_no_ifin <int> (parameters, (char*)"initSampleCount", &initSampleCount, 1, imax_value);

    CheckParam_no_ifin <int> (parameters, (char*)"initSeedCount", &initSeedCount, 1, initSampleCount);
}

pctFeatures* pctFeatures::create()
{
    auto* ptr_PCT = new pctFeatures();
    return ptr_PCT;
}

Mat pctFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    Mat imageMat = getMat(image);
    Mat descriptors;
    Ptr<PCTSignatures> featureDescriptor = PCTSignatures::create(initSampleCount, initSeedCount, pointDistribution);
    featureDescriptor->computeSignature(imageMat, descriptors);
    return descriptors.clone();
}

void pctFeatures::releaseDescriptor() {delete this;}

superpointFeatures::superpointFeatures()
{
    threshold = 0.015;
}

void superpointFeatures::setParameters(map<string, double> parameters)
{
    CheckParam_no_ifin <double> (parameters, (char*)"threshold", &threshold, 0, fmax_value);
}

superpointFeatures* superpointFeatures::create()
{
    auto* ptr_superpoint = new superpointFeatures();
    return ptr_superpoint;
}

Mat superpointFeatures::getFeatures(vector<KeyPoint>* input, string image)
{
    const char* pyName = "getSuperPointKPs";

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject* pModule = PyImport_ImportModule(pyName);
    PyErr_Print();
    PyObject* pFunc = PyObject_GetAttrString(pModule, "getSuperPointDescriptors");
    PyObject* pArgs = PyTuple_New(2);
    PyTuple_SetItem(pArgs, 0, PyBytes_FromStringAndSize(image.c_str(), Py_ssize_t(image.size())));
    PyTuple_SetItem(pArgs, 1, PyFloat_FromDouble(threshold));
    PyObject* pValue = PyObject_CallObject(pFunc, pArgs);
    PyObject* desc = PyTuple_GetItem(pValue, 1);
    long* x = (long*)PyArray_DATA(PyTuple_GetItem(PyTuple_GetItem(pValue, 0), 1));
    long* y = (long*)PyArray_DATA(PyTuple_GetItem(PyTuple_GetItem(pValue, 0), 0));
    float* descs = (float*)PyArray_DATA(desc);
    int height = PyArray_DIM(desc, 0);
    int width = PyArray_DIM(desc, 1);

    float * descsClone = new float [height*width];

    memcpy(descsClone, descs, height*width*sizeof(float));
    Mat descMat(height, width, CV_32F, descsClone);

    input->clear();
    for (int i = 0; i < height; i++)
    {
        KeyPoint buf;
        buf.pt.x = (int)x[i];
        buf.pt.y = (int)y[i];
        input->push_back(buf);
    }

    Py_DECREF(pArgs);
    Py_DECREF(pModule);
    Py_DECREF(pFunc);
    Py_DECREF(pValue);
    Mat result(height, width, CV_32F);
    descMat.copyTo(result);
    delete [] descsClone;
    PyGILState_Release(gstate);
    return result;
}

void superpointFeatures::releaseDescriptor() {delete this;}
