---
layout: default
title: Project 2 - Local Feature Matching
---

<center>
    <img src="images/proj2/notre_dame_89percent_green_correct_lines.png">
    <br>
    The top 100 most confident local feature matches from a baseline implementation of project 2. In this case, 89 were correct (lines shown in green), and 11 were incorrect (lines shown in red).
    <br><br>
</center>

# Project 2: Local Feature Matching

## Brief
* Due: 9/23/2019 11:59PM
* Project materials including writeup template [proj2.zip]()
* Hand-in: through [Canvas](https://gatech.instructure.com)
* Required files: `<your_gt_username>.zip`, `<your_gt_username>.pdf`

## Overview
The goal of this assignment is to create a local feature matching algorithm using techniques described in Szeliski chapter 4.1. The pipeline we suggest is a simplified version of the famous [SIFT](http://www.cs.ubc.ca/~lowe/keypoints/) pipeline. The matching pipeline is intended to work for _instance-level_ matching -- multiple views of the sampe physical scene.

## Setup
1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj2_env_<OS>.yml`
3. This should create an environment named 'proj1'. Activate it using the Windows command, `activate proj2` or the MacOS / Linux command, `source activate proj2`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj2_code/proj2.ipynb`
6. Ensure that all sanity checks are passing by running `pytest tests` inside the repo folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` (don't forget to submit your report, too!).

## Details
For this project, you need to implement the three major steps of a local feature matching algorithm:
* Interest point detection in `student_harris.py` (see Szeliski 4.1.1)
* Local feature description in `student_sift.py` (see Szeliski 4.1.2)
* Feature matching in `student_feature_matching.py` (see Szeliski 4.1.3)

There are numerous papers in the computer vision literature addressing each stage. For this project, we will suggest specific, relatively simple algorithms for each stage. You are encouraged to experiment with more sophisticated algorithms!

## Interest point detection (`student_harris.py`)
You will implement the Harris corner detector as described in the lecture materials and Szeliski 4.1.1. See Algorithm 4.1 in the textbook for pseudocode. The starter code gives some additional suggestions. You do not need to worry about scale invariance or keypoint orientation estimation for your baseline Harris corner detector. The original paper by Chris Harris and Mike Stephens describing their corner detector can be found [here](http://www.bmva.org/bmvc/1988/avc-88-023.pdf).

You will also implement **adaptive non-maximal suppression**. While most feature detectors simply look for local maxima in the interest function, this can lead to an uneven distribution of feature points across the image (e.g. points will be denser in regions of higher contrast). To mitigate this problem, Brown, Szeliski, and Winder (2005) only detect features that are both local maxima and whose response value is significantly (10%) greater than that of all of its neighbors within a radius _r_. The goal is to retain only those points that are the maximum in a neighborhood of radius _r_ pixels. One way to do so is to sort all points by the response strength, from large to small response. The first entry in the list is the global maximum, which is not suppressed at any radius. Then, we can iterate through the list and compute the distance to each interest point ahead of it in the list (these are pixels with even greater response strength). The minimum of distances to a keypoint's stronger neighbors (multiplying these neighbors by >=1.1 to add robustness) is the radius within which the current point is a local maximum. We call this the suppression radius of this interest point, and we save these suppression radii. Finally, we sort the suppression radii from large to small, and return the _n_ keypoints associated with the top _n_ suppression radii in this sorted order. Feel free to experiment with _n_ (we used `n=1500`).

You can read more about ANMS in [the textbook](http://szeliski.org/Book/drafts/SzeliskiBook_20100903_draft.pdf), [this conference article](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf), or [in this paper](https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf) which describes a fast variant.

## Local feature description (`student_sift.py`)
You will implement a SIFT-like local feature as described in the lecture materials and Szeliski 4.1.2. See the placeholder `get_features()` for more details. If you want to get your matching pipeline working quickly (and maybe to help debug the other algorithm stages), you might want to start with normalized patches as your features.

## Feature matching (`student_feature_matching.py`)
You will implement the "ratio test" or "nearest neighbor distance ratio test" method of matching local features as descirbed in the lecture materials and Szeliski 4.1.3. See equation 4.18 in particular. The potential matches that pass the ratio test the easiest should have a greater tendency to be correct matches -- think about _why_.

## Using the starter code (`proj2.ipynb`)
The top-level notebook provided in the starter code, `proj2.ipynb`, includes file handling, visualization, and evaluation functions for you as well as calls to placeholder versions of the three functions listed above. Running the starter code without modification will visualize random interest points matched randomly on the particular Notre Dame images shown at the top of this page. The correspondence will be visualized with `show_correspondence_circles()` and `show_correspondence_lines()` (you can comment one or both out if you prefer).

For the Notre Dame image pair, there is a ground truth evaluation in the starter code as well. `evaluate_correspondence()` will classify each match as correct or incorrect based on hand-provided matches (see `show_ground_truth_corr()` for details). The starter code also contains ground truth correspondences for two other image pairs (Mount Rushmore and Episcopal Gaudi). You can test on those images by uncommenting the appropriate lines in `proj2.ipynb`. You can create additional ground truth matches with the `CorrespondenceAnnotator().collect_ground_truth_corr()` found in `annotate_correspondences/`(but it's a tedious process).

As you implement your feature matching pipeline, you should see your performance according to `evaluate_correspondence()` increase. Hopefully you find this useful, but don't _overfit_ to the initial Notre Dame image pair, which is relatively easy. The baseline algorithm suggested here and in the starter code will give you full credit and work fairly well on these Notre Dame images, but additional image pairs provided in `extra_data.zip` are more difficult. They might exhibit more viewpoint, scale, and illumination variation. If you add enough Bells & Whistles, you should be able to match more difficult image pairs.

## Suggested implementation strategy
It is **highly suggested** that you implement the functions in this order:
* First, use `cheat_interest_points()` instead of `get_interest_points()`. This function will only work for the 3 images pairs with ground truth correspondences. This function cannot be used in your final implementation. It directly loads interest points from the ground truth correspondences for the test cases. Even with this cheating, your accuracy will initially be near zero because the starter code features are all zeros and the starter code matches are random.
* Second, change `get_features()` to return a simple feature. Start with, for instance, 16x16 patches centered on each interest point. Image patches aren't a great feature (they're not invariant to brightness changes, contrast changes, or small spatial shifts) but this is simple to implement and provides a baseline. You won't see your accuracy increase yet because the placeholder code in `match_features()` is randomly assigning matches.
* Third, implement `match_features()`. Accuracy should increase to \~40% on the Notre Dame pair if you're using 16x16 (256-dimensional) patches as your feature and if you only evaluate your 100 most confident matches. Accuracy on the other test cases will be lower (Mount Rushmore 25%, Episcopal Gaudi 7%). If you're sorting your matches by confidence (as the starter code does in `match_features()`), you should notice that your more confident matches (which pass the ratio test more easily) are more likely to be true matches.
* Fourth, finish `get_features()` by implementing a SIFT-like feature. Accuracy should increase to 70& on the Notre Dame pair, 40% on Mount Rushmore, and 15% on Episcopal Gaudi if you only evaluate your 100 most confident matches. These accuracies still aren't great because the human-selected keypoints from `cheat_interest_points()` might not match particularly well according to your feature.
* Fifth, stop using `cheat_interest_points()` and implement `get_interest_points()`. Harris corners aren't as good as ground truth points, which we know correspond, so accuracy may drop. On the other hand, you can get hundreds or even a few thousand interest points so you have more opportunities to find confident matches. If you only evaluate the most confident 100 matches (see the `num_pts_to_evaluate` paramter) on the Notre Dame pair, you should be able to achieve 90% accuracy. As long as your acuracy on the Notre Dame image pair is 80% for the 100 most confident matches, you can receive full credit for the project. When you implement adaptive non-maximal suppression your accuracy should improve even more.

You will likely need to do extra credit to get high accuracy on Mount Rushmore and Episcopal Gaudi.

**Potentially useful NumPy, OpenCV, and SciPy functions**: `np.arctan2()`, `np.sort()`, `np.reshape()`, `np.newaxis`, `np.argsort()`, `np.gradient()`, `np.histogram()`, `np.hypot()`, `np.fliplr()`, `np.flipud()`, `cv2.Sobel()`, `cv2.filter2D()`, `cv2.getGaussianKernel()`, `scipy.signal.convolve()`

**Forbidden functions** (you can use these for testing, but not in your final code): `cv2.SIFT()`, `cv2.SURF()`, `cv2.BFMatcher()`, `cv2.BFMatcher().match()`, `cv2.FlannBasedMatcher().knnMatch()`, `cv2.BFMatcher().knnMatch()`, `cv2.HOGDescriptor()`, `cv2.cornerHarris()`, `cv2.FastFeatureDetector()`, `cv2.ORB()`, `skimage.feature`, `skimage.feature.hog()`, `skimage.feature.daisy()`, `skimage.feature.corner_harris()`, `skimage.feature.corner_shi_tomasi()`, `skimage.feature.match_descriptors()`, `skimage.feature.ORB()`

We haven't enumerated all possible forbidden functions here, but using anyone else's code that performs interest point detection, feature computation, or feature matching for you is forbidden.

## Tips, tricks, and common problems
* Make sure you're not swapping _x_ and _y_ coordinates at some point. If your interest points aren't showing up where you expect or if you're getting out of bound errors, you might be swapping _x_ and _y_ coordinates. Remember, images expressed as NumPy arrays are access `image[y,x]`.
* Make sure your features aren't somehow degenerate. You can visualize your features with `plt.imshow(image1_features)`, although you may need to normalize them first. If the features are mostly zero or mostly identical, you may have made a mistake.

## Testing
We have provided a set of tests for you to evaluate your implementation. We have included tests inside `proj2.ipynb` so you can check your progress as you implement each section. When you're done with the entire project, you can call additional tests by running `pytest tests` inside the root directory of the project. _Your grade on the coding portion of the project will be further evaluated with a set of tests not provided to you._

## Writeup
For this project (and all other projects), you must do a project report using the template slides provided to you. Do <u>not</u> change the order of the slides or remove any slides, as this will affect the grading process on Gradescope and you will be deducted points. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. Then you will show and discuss the results of your algorithm. The template slides provide guidance for what you should include in your report. A good writeup doesn't just show results--it tries to draw some conclusions from the experiments. You must convert the slide deck into a PDF for your submission.

In the case of this project, show how well your matching method works not just on the Notre Dame image pair, but also on additional test cases. For the 3 iamge pairs with ground truth correspondences, you can show `eval.jpg`, which the starter code generates. For other image pairs, there is no ground truth evaluation (you can make it!) so you can show `vis_circles.jpg` or `vis_lines.jpg` instead. A good writeup will assess how important various design descisions were (e.g. by using SIFT-like features instead of normalized patches, I went from 70% good matches to 90% good matches). This is especially important if you did some of the bells & whistles and want extra credit. You should clearly demonstrate how your additions changed the behavior on particular test cases.

If you choose to do anything extra, add slides _after the slides given in the template deck_ to describe your implementation, results, and analysis. Adding slides in between the report template will cause issues with Gradescope, and you will be deducted points. You will not receive full credit for your extra credit implementations if they are not described adequately in your writeup.

## Bells & Whistles (Extra Points)
Students enrolled in 6476 are required to do 10 points worth of extra credit from the suggestions below in order to receive full credit. Extra credit beyond that can increase your grade over 100. The max score for all students is 110.

For all extra credit, be sure to include quantitative analysis showing the impact of the particular method you've implemented. Each item is "up to" some amount of points because trivial implementations may not be worthy of full extra credit.

**Interest point detection extra credit**:
* up to 5 pts: Try detecting keypoints at multiple scales or using a scale selection method to pick the best scale.
* up to 5 pts: Try estimating the orientation of keypoints to make you local features rotation invariant.
* up to 5 pts: Try the adaptive non-maximum suppression discussed in the textbook.
* up to 10 pts: Try an entirely different interest point detection strategy like that of MSER. If you implement an additional interest point detector, you can use it alone or you can take the union of keypoints detected by multiple methods.

**Local feature description extra credit**:
* up to 3 pts: The simplest thing to do is to experiment with the numerous SIFT parameters: How big should each feature be? How many local cells should it have? How many orientations should each histogram have? Different normalization schemes can have a significant effect as well. Don't get lost in parameter tuning though.
* up to 5 pts: If your keypoint detector can estimate orientation, your local feature descriptor should be built accordingly so that your pipeline is rotation invariant.
* up to 5 pts: Likewise, if you are detecting keypoints at multiple scales, you should build the features at the corresponding scales.
* up to 5 pts: Try different spatial layouts for our feature (e.g. GLOH).
* up to 10 pts: Try entirely different features (e.g. local self-similarity).
* up to 5 pts: Try spatial/geometric verification of matches (e.g. using the _x_ and _y_ locations of the features).

**Local feature matching extra credit**:
An issue with the baseline matching algorithm is the computational expense of computing distance between all pairs of features. For a reasonable implementation of the base pipeline, this is likely to be the slowest part of the code. There are numerous schemes to try and approximate or accelerate feature matching:
* up to 10 pts: Create a lower dimensional descriptor that is still accurate enough. For example, if the descriptor is 32 dimensions instead of 128, then the distance computation should be about 4 times faster. PCA would be a good way to create a low dimensional descriptor. You would need to compute the PCA basis on a sample of your local descriptors from many images.
* up to 5 pts: Use a space partitioning data structure like a kd-tree or some third-party approximate nearest neighbor package to accelerate matching.

## Rubric
* +25 pts: Harris corner detector in `student_harris.py`
* +10 pts: Non-maximal suppresion in `student_harris.py`
* +25 pts: SIFT-like local features in `student_sift.py`
* +10 pts: "ratio test" matching in `student_feature_matching.py`
* +20 pts: Report with several examples of local feature matching.
* +10 pts: Extra credit (up to ten points). You are welcome to implement any bells & whistles. There is no particular mandatory extra credit for the graduate section for this project. Instead, graduate students can pick any 10 points of extra credit to get full credit (and do 10 further extra credit points to get a max score of 110).
* -5\*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format.

## Submission Format
This is very important as you will lose 5 points for every time you do not follow the instructions. You will attach two items in your submission on Canvas:

1. `<your_gt_username>.zip` containing:
    * `proj2_code/` - directory containing all your code for this assignment
    * `additional_data/` - (optional) if you use any data other than the images we provide you, please include them here
    * `README.txt` - (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g. any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj1.pdf` - your report

Do <u>not</u> install any additional packages inside the conda environment. The TAs will use the same environment as defined in the config files we provide you, so anything that's not in there by default will probably cause your code to break during grading. Do <u>not</u> use absolute paths in your code or your code will break. Use relative paths like the starter code already does. Failure to follow any of these instructions will lead to point deductions. Create the zip file using `python zip_submission.py --gt_username <your_gt_username>` (it will zip up the appropriate directories/files for you!) and hand it in with your PDF through Canvas.

## Credits
Assignment developed by John Lambert, Patsorn Sangkloy, Vijay Upadhya, Cusuh Ham, Frank Dellaert, and James Hays based on a similar project by Derek Hoiem.
