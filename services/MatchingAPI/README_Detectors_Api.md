# Implemented detectors and their parameters

## ORB detector
Parameters:

**nfeatures**   The maximum number of features to retain. Positive integer number.

**scaleFactor** Pyramid decimation ratio, float number between 1 and 2. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically.

**nlevels** The number of pyramid levels. Positive integer between 1 and 8.

**edgeThreshold**   This is size of the border where the features are not detected. Integer number between 0 and 255

**firstLevel**  The level of pyramid to put source image to. Currently only 0 value supported.

## KAZE detector
Parameters:

**threshold**   Detector response threshold to accept point. Float number between 0 and 1. Values closer to 0 give more detections.

**nOctaves**	Maximum octave evolution of the image. Positive integer from 1 to 20. Higher numbers of nOctaves cause more costly computations.

**nOctaveLayers**	Default number of sublevels per scale level. Positive integer between 1 and 20. Higher numbers of nOctaves cause more costly computations.

**diffusivity** Diffusivity type. DIFF_PM_G1(0), DIFF_PM_G2(1), DIFF_WEICKERT(2) or DIFF_CHARBONNIER(3)

## AKAZE detector
Parameters:

**threshold**   Detector response threshold to accept point. Float number between 0 and 1. Values closer to 0 give more detections.

**nOctaves**	Maximum octave evolution of the image. Positive integer from 1 to 20. Higher numbers of nOctaves cause more costly computations.

**nOctaveLayers**	Default number of sublevels per scale level. Positive integer between 1 and 20. Higher numbers of nOctaves cause more costly computations.

**diffusivity** Diffusivity type. DIFF_PM_G1 = 0, DIFF_PM_G2 = 1, DIFF_WEICKERT = 2 or DIFF_CHARBONNIER = 3.

## AGAST detector
Parameters:

**threshold**   This is threshold for a gradient magnitude where the features are not detected. Integer number between 0 and 255

**nonmaxSuppression**   Boolean flag. Turns on or off non-max suppression processing. Dramatically affects performance.

**type**    Detector type. AGAST_5_8 = 0, AGAST_7_12d = 1, AGAST_7_12s = 2, OAST_9_16 = 3.

## GFT (GFTT) detector
Parameters:

**maxCorners**  Maximum number of detected corners. Positive integer number.

**qualityLevel**    Quality of response from detected point. Positive double precision number.

**minDistance** Minimum distance between detected points. Positive double precision number.

**blockSize**   Number of blocks used to compute a response of point. Positive integer number.

**useHarrisDetector**   Boolean flag for whether or not to use Harris detector.

**k**   Free parameter of the Harris detector. Positive double precision number.

## MSER detector
Parameters:

**_delta**	It compares (sizei−sizei−delta)/sizei−delta

**_min_area**	Prune the area which smaller than minArea. Positive number.

**_max_area**	Prune the area which bigger than maxArea. Positive number.

**_max_variation**	Prune the area have similar size to its children. Positive number.

**_min_diversity**	For color image, trace back to cut off mser with diversity less than min_diversity. Positive number.

**_max_evolution**	For color image, the evolution steps. Positive number.

**_area_threshold**	For color image, the area threshold to cause re-initialize. Positive number.

**_min_margin** For color image, ignore too small margin. Positive number.

**_edge_blur_size**	For color image, the aperture size for edge blur. Positive number.

## BRISK detector
Parameters:

**thresh**  AGAST detection threshold score. Positive integer from 0 to 255.

**octaves** Detection octaves. Use 0 to do single scale. Positive integer from 0 to 9.

## FAST detector
Parameters:

**threshold**   Detection threshold score. Positive integer from 0 to 255.

**nonmaxSuppression**   Boolean flag. Turns on or off non-max suppression processing. Dramatically affects performance.

**type**    Detector type.  TYPE_5_8 = 0, TYPE_7_12 = 1, TYPE_9_16 = 2.

## BLOB detector
Parameters:

No parameters available. TBA.

## STAR(CenSurE) detector
Parameters:

**maxSize** Some parameter. Influences on amount and quality of points. Positive integer. Actual range is [1 20], but it can vary depending on image conditions.

**responseThreshold**   Detector response threshold to accept point. Positive integer.

**lineThresholdProjected**  Yet another threshold. Positive integer.

**lineThresholdBinarized**  Yet another threshold. Positive integer.

**suppressNonmaxSize**  Parameter, controlling non-max suppression level. Positive integer between 0 and 39.

## MSDS detector
Parameters:

**m_patch_radius**  Radius of a patch. Integer number between 0 and 100

**m_search_area_radius**  Search radius for response. Integer number between 1 and 100

**m_nms_radius**    Non-max suppression radius. Positive integer.

**m_nms_scale_radius**   Non-max suppression scale radius. Integer number between 0 and 1

**m_th_saliency**   Saliency parameter. Floating point number.

**m_kNN**   Number of nearest neighbours. Dramatically affects performance. Integer number Integer number between 1 and 100

**m_scale_factor**  Pyramid scale factor. Positive real number between 1.01 and 11.0

**m_n_scales**  Number of layers in pyramid. Integer number between 1 and 10 or -1.

**m_compute_orientation**   Boolean flag. Allows computation of an orientation.

## HLFD detector
Parameters:

**numOctaves**  Positive integer greater than 1.

**corn_thresh** Corner detection threshold. Negative numbers increase amount of detected points, positive numbers decrease ones.

**DOG_thresh**  DoG response threshold.

**maxCorners**  Maximum number of detected points.

**num_layers**  Takes one of the two possible values: 2 or 4.
