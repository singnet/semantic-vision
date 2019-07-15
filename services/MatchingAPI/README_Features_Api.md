# Implemented feature methods and their parameters

## ORB features
Parameters:

**WTA_K**. Possible values - [2, 3, 4]. Type int. The number of points that produce each element of the oriented BRIEF descriptor. The
default value 2 means the BRIEF where we take a random point pair and compare their brightnesses,
so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3
random points (of course, those point coordinates are random, but they are generated from the
pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel
rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such
output will occupy 2 bits, and therefore it will need a special variant of Hamming distance,
denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each
bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).

## KAZE features
Parameters:

**extended**. Possible values - [0, 1]. Type bool. Set to enable extraction of extended (128-byte) descriptor.

**upright**. Possible values - [0, 1]. Type bool. Set to enable use of upright descriptors (non rotation-invariant).

**diffusivityType**. Possible values - [0, 1, 2, 3].
Type int. Diffusivity type. 0 - KAZE::DIFF_PM_G1, 1 - KAZE::DIFF_PM_G2, 2 - KAZE::DIFF_WEICKERT, 3 - KAZE::DIFF_CHARBONNIER

## AKAZE features
Parameters:

**descriptor_type**. Possible values - [2, 3, 4, 5]. Type enum. Type of the extracted descriptor.
2 - AKAZE::DESCRIPTOR_KAZE, 3 - AKAZE::DESCRIPTOR_KAZE_UPRIGHT, 4 - AKAZE::DESCRIPTOR_MLDB, 5 - AKAZE::DESCRIPTOR_MLDB_UPRIGHT

**descriptor_size**. Possible values - [0 - 162*descriptor_channels] (for example, descriptor_channels = 3, then max descriptor_size is equal to 162*3 = 486). Type int. Size of the descriptor in bits. 0 - full size.

**diffusivityType**. Possible values - [0, 1, 2, 3]. Type int. Diffusivity type.
0 - KAZE::DIFF_PM_G1, 1 - KAZE::DIFF_PM_G2, 2 - KAZE::DIFF_WEICKERT, 3 - KAZE::DIFF_CHARBONNIER

**descriptor_channels**. Possible values - [1, 2, 3]. Type int. Number of channels in the descriptor.

## BRISK Features
Parameters:

**patternScale**. Possible values - [1.0 - max_float_value]. High values are not recommended. Type float.
Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

## BRIEF Features
Parameters:

**bytes**. Possible values - [16, 32, 64]. Type int. Legth of the descriptor in bytes.

**use_orientation**. Possible values - [0, 1]. Type bool. Sample patterns using keypoints orientation.

## FREAK Features
Parameters:

**orientationNormalized**. Possible values - [0, 1]. Type bool. Enable orientation normalization.

**scaleNormalized**. Possible values - [0, 1]. Type bool. Enable scale normalization.

**patternScale**. Possible values - [0, max_float_value]. Type float. Scaling of the description pattern.

**nOctaves**. Possible values - [0, max_int_value]. Type int. Number of octaves covered by the detected keypoints.

## LUCID Features
Parameters:

**lucid_kernel**. Possible values - [0, max_int_value]. Type int. Higher value - worse performance. Kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth.

**blur_kernel**. Possible values - [0, max_int_value]. Type int. If value is too high - matching will degrade. Kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth.

## LATCH Features
Parameters:

**bytes**. Possible values = [64, 32, 16, 8, 4, 2 or 1]. Type int. Size of the descriptor.

**rotationInvariance**. Possible values - [0, 1]. Type bool. Whether or not the descriptor should compensate for orientation changes.
It makes sense only to use "true" with detectors which computes patch orientation. Though it will work with any.

**half_ssd_size** Possible values - [0 - max_int_value]. Type int. The size of half of the mini-patches size.
For example, if we would like to compare triplets of patches of size 7x7 then the half_ssd_size should be (7-1)/2 = 3.

**sigma**. Possible values - [0 - max_double_value]. Type double. Sigma value for GaussianBlur smoothing of the source image.
Source image will be used without smoothing in case sigma value is 0.

## DAISY Features
Parameters:

**radius**. Possible values - [0 - max_float_value]. Type float. High values are not recommended. Radius of the descriptor
at the initial scale

**q_radius**. Possible values - [1 - 656]. Type int. Amount of radial range division quantity

**q_theta**. Possible values - [0 - max_int_value]. Type int. High values are not recommended. Amount of angular range
division quantity

**q_hist**. Possible values - [1 - max_int_value]. Type int. High values are not recommended. Amount of gradient
orientations range division quantity

**norm**. Possible values - [100, 101, 102, 103] Type enum. Choose descriptors normalization type, where
100 - DAISY::NRM_NONE will not do any normalization (default),
101 - DAISY::NRM_PARTIAL mean that histograms are normalized independently for L2 norm equal to 1.0,
102 - DAISY::NRM_FULL mean that descriptors are normalized for L2 norm equal to 1.0,
103 - DAISY::NRM_SIFT mean that descriptors are normalized for L2 norm equal to 1.0 but no individual one is bigger than
0.154 as in SIFT

**interpolation**. Possible values - [0, 1]. Type bool. Switch to disable interpolation for speed improvement at
minor quality loss

**use_orientation**. Possible values - [0, 1]. Type bool. Sample patterns using keypoints orientation.

## VGG Features
Parameters:

**desc**. Possible values - [100, 101, 102, 103]. Type enum. Type of descriptor to use, VGG::VGG_120 is default (120 dimensions float).
100 - VGG::VGG_120, 101 - VGG::VGG_80, 102 - VGG::VGG_64, 103 - VGG::VGG_48

**isigma**. Possible values - [1 - 1000]. High value is not recommended. Type float. Gaussian kernel value for image blur.

**img_normalize** Possible values - [0, 1]. Type bool. Use image sample intensity normalization.

**use_scale_orientation**. Possible values - [0, 1]. Type bool. Sample patterns using keypoints orientation.

**scale_factor**. Possible values - [1 - float_max_value]. Type float. Adjust the sampling window of detected keypoints to 64.0f (VGG sampling window)
Recommended values:
6.25f is default and fits for KAZE, SURF detected keypoints window ratio
6.75f should be the scale for SIFT detected keypoints window ratio
5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
0.75f should be the scale for ORB keypoints ratio

**dsc_normalize**. Possible values - [0, 1]. Type bool. Clamp descriptors to 255 and convert to uchar CV_8UC1

## BOOST Features
Parameters:

**desc**. Possible values - [100, 101, 102, 200, 300, 301, 302]. Type enum. Type of descriptor to use.
100 - BoostDesc::BGM, 101 - BoostDesc::BGM_HARD, 102 - BoostDesc::BGM_BILINEAR, 200 - BoostDesc::LBGM,
300 - BoostDesc::BINBOOST_64, 301 - BoostDesc::BINBOOST_128, 302 - BoostDesc::BINBOOST_256

**use_orientation**. Possible values - [0, 1]. Type bool. Sample patterns using keypoints orientation.

**scale_factor**. Possible values - [1 - float_max_value]. Type float. Adjust the sampling window of detected keypoints.
Recommended values:
6.25f is default and fits for KAZE, SURF detected keypoints window ratio
6.75f should be the scale for SIFT detected keypoints window ratio
5.00f should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints window ratio
0.75f should be the scale for ORB keypoints ratio
1.50f was the default in original implementation

## PCT Features
Parameters:

**initSampleCount**. Possible values - [1 - max_int_value]. Higher values - worse performance. Type int. Number of points used for image sampling.

**initSeedCount**.  Possible values - [1-max_int_value]. Type int. Number of initial clusterization seeds. Must be lower or equal to initSampleCount

**pointDistribution**. Possible values - [0, 1, 2]. Type enum. Distribution of generated points.
0 - PointDistribution::UNIFORM, 1 - PointDistribution::REGULAR, 2 - PointDistribution::NORMAL
 
## Superpoint features

**threshold** Affects on detector which will detect keypoints to be descripted. Can't be stacked with ANY other detectors since Superpoint uses it's own keypoints.
The higher threshold is, the less points you'll get. Double. Possible values - [0 - max_double_value]. Default value is 0.015.
