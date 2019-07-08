#include "Transformations.h"

ITransform* ChooseTransform(const char *name)
{
    if (strncmp(name, (char*)"Affine", 6) == 0)
    {
        cout << "Getting affine transform parameters" << endl;
        return AffineTransform::create();
    }
    else if (strncmp(name, (char*)"Perspective", 11) == 0)
    {
        cout << "Getting perspective transform parameters" << endl;
        return PerspectiveTransform::create();
    }
    else if (strncmp(name, (char*)"Essential", 9) == 0)
    {
        cout << "Getting essential matrix" << endl;
        return EssentialMatrix::create();
    }
    else if (strncmp(name, (char*)"Fundamental", 11) == 0)
    {
        cout << "Getting fundamental matrix" << endl;
        return FundamentalMatrix::create();
    }
    else if (strncmp(name, (char*)"Homography", 10) == 0)
    {
        cout << "Getting homography matrix" << endl;
        return Homography::create();
    }
    else if (strncmp(name, (char*)"Similarity", 10) == 0)
    {
        cout << "Getting similarity transform parameters" << endl;
        return Similarity::create();
    }
    else if (strncmp(name, (char*)"ShiftScale", 10) == 0)
    {
        cout << "Getting ShiftScale transform parameters" << endl;
        return ShiftScale::create();
    }
    else if (strncmp(name, (char*)"ShiftRot", 8) == 0)
    {
        cout << "Getting ShiftRot transform parameters" << endl;
        return ShiftRot::create();
    }
    else if (strncmp(name, (char*)"Shift", 5) == 0)
    {
        cout << "Getting shift transform parameters" << endl;
        return Shift::create();
    }
    else if (strncmp(name, (char*)"Bilinear", 8) == 0)
    {
        cout << "Getting bilinear transform parameters" << endl;
        return Bilinear::create();
    }
    else if (strncmp(name, (char*)"Polynomial", 10) == 0)
    {
        cout << "Getting polynomial transform parameters" << endl;
        return Poly::create();
    }
    else
    {
        cout << "No transform was chosen or there is a mistake in the name. Using homography by default" << endl;
        return Homography::create();
    }
}

AffineTransform::AffineTransform() = default;

void AffineTransform::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

AffineTransform* AffineTransform::create()
{
    auto* ptr_detector = new AffineTransform();
    return ptr_detector;
}

vector<double> AffineTransform::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_3_only, second_3_only;
    for (int i = 0; i < 3; i++)
    {
        first_3_only.push_back(first[matches[i].queryIdx].pt);
        second_3_only.push_back(second[matches[i].trainIdx].pt);
    }
    Mat affParams = getAffineTransform(first_3_only, second_3_only);
    for (int i = 0; i < affParams.cols*affParams.rows; i++)
    {
        transformParameters.push_back(affParams.at<double>(i));
    }
    return transformParameters;
}

void AffineTransform::releaseTransform() {delete this;}

PerspectiveTransform::PerspectiveTransform() = default;

void PerspectiveTransform::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

PerspectiveTransform* PerspectiveTransform::create()
{
    auto* ptr_detector = new PerspectiveTransform();
    return ptr_detector;
}

vector<double> PerspectiveTransform::getTransform(vector<KeyPoint> first, vector<KeyPoint> second,
        vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_4_only, second_4_only;
    for (int i = 0; i < 4; i++)
    {
        first_4_only.push_back(first[matches[i].queryIdx].pt);
        second_4_only.push_back(second[matches[i].trainIdx].pt);
    }
    Mat perspParams = getPerspectiveTransform(first_4_only, second_4_only);
    for (int i = 0; i < perspParams.cols*perspParams.rows; i++)
    {
        transformParameters.push_back(perspParams.at<double>(i));
    }
    return transformParameters;
}

void PerspectiveTransform::releaseTransform() {delete this;}

EssentialMatrix::EssentialMatrix()
{
    focal = 1.0;
    pp_x = 0.0;
    pp_y = 0.0;
    method = cv::LMEDS;
    prob = 0.999;
    threshold = 1.0;
    methodNames[8] = (char*)"RANSAC";
    methodNames[4] = (char*)"LMEDS";
}

void EssentialMatrix::setParameters(map<string, double> params)
{
    double dmax = std::numeric_limits<double>::max();
    double dmin = std::numeric_limits<double>::min();

    CheckParam_no_ifin<double> (params, (char*)"focal", &focal, dmin, dmax);

    CheckParam_no_ifin<double> (params, (char*)"pp_x", &pp_x, dmin, dmax);

    CheckParam_no_ifin<double> (params, (char*)"pp_y", &pp_y, dmin, dmax);

    int range[] = {4, 8};
    CheckParam<int> (params, (char*)"method", &method, range, 2);
    cout << " - " << methodNames[method];

    CheckParam_no_ifin<double> (params, (char*)"prob", &prob, 1e-308, 1 - 1e-308);

    CheckParam_no_ifin<double> (params, (char*)"threshold", &threshold, dmin, dmax);
}

vector<double> EssentialMatrix::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_kps_filtered, second_kps_filtered;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered.push_back(first[oneMatch.queryIdx].pt);
        second_kps_filtered.push_back(second[oneMatch.trainIdx].pt);
    }

    Mat essMat = findEssentialMat(first_kps_filtered, second_kps_filtered, focal, Point2d(pp_x,pp_y), method, prob, threshold);
    for (int i = 0; i < essMat.cols*essMat.rows; i++)
    {
        transformParameters.push_back(essMat.at<double>(i));
    }
    return transformParameters;
}

EssentialMatrix* EssentialMatrix::create()
{
    auto* ptr_detector = new EssentialMatrix();
    return ptr_detector;
}

void EssentialMatrix::releaseTransform() {delete this;}

FundamentalMatrix::FundamentalMatrix()
{
    methodNames[1] = (char*)"FM_7POINT";
    methodNames[2] = (char*)"FM_8POINT";
    methodNames[4] = (char*)"FM_LMEDS";
    methodNames[8] = (char*)"FM_RANSAC";

    method = FM_RANSAC;
    ransacReprojThreshold = 3.;
    confidence = 0.99;
}

void FundamentalMatrix::setParameters(map<string, double> params)
{
    double dmax = std::numeric_limits<double>::max();
    double dmin = std::numeric_limits<double>::min();

    int range[] = {1, 2, 4, 8};
    CheckParam<int> (params, (char*)"method", &method, range, 4);
    cout << " - " << methodNames[method];

    CheckParam_no_ifin<double> (params, (char*)"ransacReprojThreshold", &ransacReprojThreshold, dmin, dmax);

    CheckParam_no_ifin<double> (params, (char*)"confidence", &confidence, 1e-308, 1 - 1e-308);
}

vector<double> FundamentalMatrix::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_kps_filtered, second_kps_filtered;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered.push_back(first[oneMatch.queryIdx].pt);
        second_kps_filtered.push_back(second[oneMatch.trainIdx].pt);
    }

    Mat fundMat = findFundamentalMat(first_kps_filtered, second_kps_filtered, method, ransacReprojThreshold, confidence);
    for (int i = 0; i < fundMat.cols*fundMat.rows; i++)
    {
        transformParameters.push_back(fundMat.at<double>(i));
    }
    return transformParameters;
}

FundamentalMatrix* FundamentalMatrix::create()
{
    auto* ptr_detector = new FundamentalMatrix();
    return ptr_detector;
}

void FundamentalMatrix::releaseTransform() {delete this;}

Homography::Homography()
{
    methodNames[0] = (char*)"Least Squares";
    methodNames[4] = (char*)"LMEDS";
    methodNames[8] = (char*)"RANSAC";
    methodNames[16] = (char*)"RHO";
    method = 0;
    ransacReprojThreshold = 3;
    maxIters = 2000;
    confidence = 0.995;
}

void Homography::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();
    double dmax = std::numeric_limits<double>::max();
    double dmin = std::numeric_limits<double>::min();

    int range[] = {0, 4, 8, 16};
    CheckParam<int> (params, (char*)"method", &method, range, 4);
    cout << " - " << methodNames[method];

    CheckParam_no_ifin<double> (params, (char*)"ransacReprojThreshold", &ransacReprojThreshold, dmin, dmax);

    CheckParam_no_ifin<int> (params, (char*)"maxIters", &maxIters, 1, imax);

    CheckParam_no_ifin<double> (params, (char*)"confidence", &confidence, 1e-308, 1 - 1e-308);
}

vector<double> Homography::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_kps_filtered, second_kps_filtered;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered.push_back(first[oneMatch.queryIdx].pt);
        second_kps_filtered.push_back(second[oneMatch.trainIdx].pt);
    }

    Mat homoMat = findHomography(first_kps_filtered, second_kps_filtered, method, ransacReprojThreshold, noArray(), maxIters, confidence);
    for (int i = 0; i < homoMat.cols*homoMat.rows; i++)
    {
        transformParameters.push_back(homoMat.at<double>(i));
    }
    return transformParameters;
}

Homography* Homography::create()
{
    auto* ptr_detector = new Homography();
    return ptr_detector;
}

void Homography::releaseTransform() {delete this;}

Similarity::Similarity()
{
    methodNames[4] = (char*)"LMEDS";
    methodNames[8] = (char*)"RANSAC";

    method = 8;
    ransacReprojThreshold = 3;
    maxIters = 2000;
    confidence = 0.99;
    refineIters = 10;
}

void Similarity::setParameters(map<string, double> params)
{
    int imax = std::numeric_limits<int>::max();
    double dmax = std::numeric_limits<double>::max();
    double dmin = std::numeric_limits<double>::min();

    int range[] = {4, 8};
    CheckParam<int> (params, (char*)"method", &method, range, 4);
    cout << " - " << methodNames[method];

    CheckParam_no_ifin<double> (params, (char*)"ransacReprojThreshold", &ransacReprojThreshold, dmin, dmax);

    CheckParam_no_ifin<double> (params, (char*)"confidence", &confidence, 1e-308, 1 - 1e-308);

    CheckParam_no_ifin<int> (params, (char*)"maxIters", &maxIters, 1, imax);

    CheckParam_no_ifin<int> (params, (char*)"refineIters", &refineIters, 0, imax);
}

vector<double> Similarity::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> transformParameters;
    vector<Point2f> first_kps_filtered, second_kps_filtered;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered.push_back(first[oneMatch.queryIdx].pt);
        second_kps_filtered.push_back(second[oneMatch.trainIdx].pt);
    }

    Mat similarity = estimateAffinePartial2D(first_kps_filtered, second_kps_filtered, noArray(), method,
            ransacReprojThreshold,  maxIters, confidence, refineIters);
    for (int i = 0; i < similarity.cols*similarity.rows; i++)
    {
        transformParameters.push_back(similarity.at<double>(i));
    }
    return transformParameters;
}

Similarity* Similarity::create()
{
    auto* ptr_detector = new Similarity();
    return ptr_detector;
}

void Similarity::releaseTransform() {delete this;}

Shift::Shift() = default;

void Shift::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

vector<double> Shift::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y;
    vector<double> transformParameters;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered_x.push_back(first[oneMatch.queryIdx].pt.x);
        second_kps_filtered_x.push_back(second[oneMatch.trainIdx].pt.x);

        first_kps_filtered_y.push_back(first[oneMatch.queryIdx].pt.y);
        second_kps_filtered_y.push_back(second[oneMatch.trainIdx].pt.y);
    }

    findShift(first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y, &transformParameters);
    return transformParameters;

}

Shift* Shift::create()
{
    auto* ptr_detector = new Shift();
    return ptr_detector;
}

void Shift::releaseTransform() {delete this;}

ShiftScale::ShiftScale() = default;

void ShiftScale::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

vector<double> ShiftScale::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y;
    vector<double> transformParameters;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered_x.push_back(first[oneMatch.queryIdx].pt.x);
        second_kps_filtered_x.push_back(second[oneMatch.trainIdx].pt.x);

        first_kps_filtered_y.push_back(first[oneMatch.queryIdx].pt.y);
        second_kps_filtered_y.push_back(second[oneMatch.trainIdx].pt.y);
    }

    findShiftScale(first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y, &transformParameters);
    return transformParameters;
}

ShiftScale* ShiftScale::create()
{
    auto* ptr_detector = new ShiftScale();
    return ptr_detector;
}

void ShiftScale::releaseTransform() {delete this;}

ShiftRot::ShiftRot() = default;

void ShiftRot::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

vector<double> ShiftRot::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y;
    vector<double> transformParameters;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered_x.push_back(first[oneMatch.queryIdx].pt.x);
        second_kps_filtered_x.push_back(second[oneMatch.trainIdx].pt.x);

        first_kps_filtered_y.push_back(first[oneMatch.queryIdx].pt.y);
        second_kps_filtered_y.push_back(second[oneMatch.trainIdx].pt.y);
    }

    findShiftRot(first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y, &transformParameters);
    return transformParameters;
}

ShiftRot* ShiftRot::create()
{
    auto* ptr_detector = new ShiftRot();
    return ptr_detector;
}

void ShiftRot::releaseTransform() {delete this;}

Poly::Poly() = default;

void Poly::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

vector<double> Poly::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y;
    vector<double> transformParameters;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered_x.push_back(first[oneMatch.queryIdx].pt.x);
        second_kps_filtered_x.push_back(second[oneMatch.trainIdx].pt.x);

        first_kps_filtered_y.push_back(first[oneMatch.queryIdx].pt.y);
        second_kps_filtered_y.push_back(second[oneMatch.trainIdx].pt.y);
    }

    bool successful_flag = findPoly(first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y, &transformParameters, false);
    if (!successful_flag)
        cout << endl << "zero determinant. No parameters could be found" << endl;

    return transformParameters;
}

Poly* Poly::create()
{
    auto* ptr_detector = new Poly();
    return ptr_detector;
}

void Poly::releaseTransform() {delete this;}

Bilinear::Bilinear() = default;

void Bilinear::setParameters(map<string, double> params)
{
    cout << endl << "No parameters needed for that transform" << endl;
}

vector<double> Bilinear::getTransform(vector<KeyPoint> first, vector<KeyPoint> second, vector<DMatch> matches)
{
    vector<double> first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y;
    vector<double> transformParameters;
    for (auto& oneMatch : matches)
    {
        first_kps_filtered_x.push_back(first[oneMatch.queryIdx].pt.x);
        second_kps_filtered_x.push_back(second[oneMatch.trainIdx].pt.x);

        first_kps_filtered_y.push_back(first[oneMatch.queryIdx].pt.y);
        second_kps_filtered_y.push_back(second[oneMatch.trainIdx].pt.y);
    }

        bool successful_flag = findPoly(first_kps_filtered_x, first_kps_filtered_y, second_kps_filtered_x, second_kps_filtered_y, &transformParameters, true);
    if (!successful_flag)
        cout << endl << "zero determinant. No parameters could be found" << endl;

    return transformParameters;
}

Bilinear* Bilinear::create()
{
    auto* ptr_detector = new Bilinear();
    return ptr_detector;
}

void Bilinear::releaseTransform() {delete this;}