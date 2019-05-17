# Implemented transforms and their parameters

## Affine
Getting affine transform parameters. Input parameters:

**No parameters**

## Perspective
Getting affine transform parameters. Input parameters:

**No parameters**

## Essential
Getting essential matrix. Input parameters:

**focal**
Type double. Possible values - [double_min - double_max] focal length of the camera. Note that this function assumes that keypoints_first and keypoints_second
are feature points from cameras with same focal length and principal point.

**pp_x**
Type double. Possible values - [double_min - double_max]. Principal point of the camera, x-value

**pp_y**
Type double. Possible values - [double_min - double_max]. Principal point of the camera, y-value

**method**
Type enum. Possible values - [4, 8]. Method for computing a fundamental matrix. 4 - for the LMedS algorithm. 8 - for 
the RANSAC algorithm.

**prob**
Type double. Possible values - [1e-308 - 1-1e-308].  Parameter used for the RANSAC or LMedS methods. It specifies 
a desirable level of confidence (probability) that the estimated matrix is correct.

**threshold**
Type double. Parameter used for RANSAC. It is the maximum distance from a 
point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for computing the
final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point localization, 
image resolution, and the image noise.

## Fundamental
Getting Fundamental matrix. Input parameters:

**method**
Type enum. Possible values - [1, 2, 4, 8]. Method for computing a fundamental matrix. 1 - FM_7POINT for a 7-point 
algorithm; 2 -  FM_8POINT for an 8-point algorithm; 4 - FM_LMEDS for the LMedS algorithm; 8 - FM_RANSAC for the RANSAC 
algorithm.
 
**ransacReprojThreshold**
Type double. Parameter used only for RANSAC. It is the maximum distance 
from a point to an epipolar line in pixels, beyond which the point is considered an outlier and is not used for 
computing the final fundamental matrix. It can be set to something like 1-3, depending on the accuracy of the point 
localization, image resolution, and the image noise.

**confidence**
Type double. Possible values - [1e-308 - 1-1e-308]. Parameter used for the RANSAC and LMedS methods only. It specifies a desirable level
of confidence (probability) that the estimated matrix is correct.

## Homography
Getting homography matrix. Input parameters:

**method**
Type enum. Possible values - [0, 4, 8, 16]. Method used to compute a homography matrix. 0 - a regular method using all 
the points, i.e., the least squares method; 4 - Least-Median robust method; 8 - RANSAC-based robust method; 16 - 
PROSAC-based robust method.

**ransacReprojThreshold**
Type double. Maximum allowed reprojection error to treat a point pair as an inlier (used in the RANSAC and RHO methods 
only). If keypoints are measured in pixels, it usually makes sense to set this parameter somewhere in the range of 1 to 
10.

**maxIters**
Type int. Possible values - [1 - int_max]. High values are not recommended due to low performance. The maximum number of 
RANSAC iterations.

**confidence**
Type double. Possible values - [1e-308 - 1-1e-308]. Confidence level.

## Similarity
Getting parameters of similarity transformations. Input parameters:

**method**
Type enum. Possible values - [4, 8]. Robust method used to compute transformation. 4 - Least-Median robust method;
8 - RANSAC-based robust method.

**ransacReprojThreshold**
Type double. Possible values - [double_min - double_max]. Maximum reprojection error in the RANSAC algorithm to consider
a point as an inlier. Applies only to RANSAC.

**maxIters**
Type int. Possible values - [1 - int_max]. High values are not recommended. The maximum number of robust method
iterations.

**confidence**
Type double. Possible values - [1e-308 - 1-1e-308]. Confidence level, between 0 and 1, for the estimated transformation. 
Anything between 0.95 and 0.99 is usually good enough. Values too close to 1 can slow down the estimation significantly.
Values lower than 0.8-0.9 can result in an incorrectly estimated transformation.

**refineIters**
Type int. Possible values - [0 - int_max]. Maximum number of iterations of refining algorithm (Levenberg-Marquardt).
Passing 0 will disable refining, so the output matrix will be output of robust method.

## ShiftScale
Getting parameters of shift and scale transformations between keypoints / images. No parameters needed.

## ShiftRot
Getting parameters of shift and rotation transformations between keypoints / images. No parameters needed.

## Shift
Getting parameters of shift transformation between keypoints / images. No parameters needed.

## Bilinear
Getting bilinear transformation which fits best to the given set of reference points / images. No parameters needed.

## Polynomial
Getting polynomial transformation which fits best to the given set of reference points / images. No parameters needed.