---
layout: default
title: Project 4 - Depth Estimation using Stereo
---

# Project 4: Depth Estimation using Stereo

## Brief
* Due:
  * 10/28/2019 11:59PM - Part 1 code only
  * 11/04/2019 11:59PM - Part 2 code + full report
* Project materials: [proj4_part1_v1.zip](projects/proj4_part1_v1.zip), part 2 coming soon
* Hand-in: through [Canvas](https://gatech.instructure.com) AND [Gradescope](https://www.gradescope.com)
* Required files:
  * Intermediate: `<your_gt_username>.zip` on Canvas
  * Final: `mc-cnn.ipynb` on Canvas, `<your_gt_username>_proj4.pdf` on Gradescope

## Setup
Note that the proj3 environment should work for this project! If you run into import module errors, try `pip install -e .` again, and if that still doesn’t work, you may have to create a fresh environment.

1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj4_env_<OS>.yml`
3. This should create an environment named 'proj4'. Activate it using the Windows command, `activate proj4` or the MacOS / Linux command, `source activate proj4`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj4_code/simple_stereo.ipynb`
6. Ensure that all sanity checks are passing by running `pytest` inside the "unit_tests/" folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` and submit to Canvas (don't forget to submit your report to Gradescope!).

## Project Description
We have split the description for this project into two parts:

1. [Part 1](proj4_part1.md)
2. Part 2 (**will be released soon**)


## Writeup
For this project (and all other projects), you must do a project report using the template slides provided to you. Do <u>not</u> change the order of the slides or remove any slides, as this will affect the grading process on Gradescope and you will be deducted points. In the report you will describe your algorithm and any decisions you made to write your algorithm a particular way. Then you will show and discuss the results of your algorithm. The template slides provide guidance for what you should include in your report. A good writeup doesn't just show results--it tries to draw some conclusions from the experiments. You must convert the slide deck into a PDF for your submission.

If you choose to do anything extra, add slides _after the slides given in the template deck_ to describe your implementation, results, and analysis. Adding slides in between the report template will cause issues with Gradescope, and you will be deducted points. You will not receive full credit for your extra credit implementations if they are not described adequately in your writeup.

## Rubric
* +60 pts: Code
  * Part 1:
    * 10 pts: `generate_random_stereogram` in `utils.py`
    * 10 pts: `similarity_measures.py`
    * 20 pts: `disparity_map.py`
  * Part 2:
    * 20 pts: `mc-cnn.ipynb`
* +40 pts: Report
  * 20 pts: part 1 analyses
  * 20 pts: part 2 analyses
* -5\*n pts:  Lose 5 points for every time you do not follow the instructions for the hand-in format.

## Submission Format
This is very important as you will lose 5 points for every time you do not follow the instructions. You will have two submission files for this project:

* 10/28/2019 intermediate submission:
  1. `<your_gt_username>.zip` via Canvas containing:
    * `proj4_code/` - directory containing all the code for part 1
    * `additional_data/` - (optional) if you use any data other than the images we provide you, please include them here
    * `README.txt` - (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g. any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
* 11/04/2019 final submission:
  1. `mc-cnn.ipynb` via Canvas - your part 2 code
  2. `<your_gt_username>_proj4.pdf` via Gradescope - your report

Do <u>not</u> install any additional packages inside the conda environment. The TAs will use the same environment as defined in the config files we provide you, so anything that's not in there by default will probably cause your code to break during grading. Do <u>not</u> use absolute paths in your code or your code will break. Use relative paths like the starter code already does. Failure to follow any of these instructions will lead to point deductions. Create the zip file using `python zip_submission.py --gt_username <your_gt_username>` (it will zip up the appropriate directories/files for you!) and hand it in through Canvas. Remember to submit your report as a PDF to Gradescope as well.

## Credit
Assignment developed by Ayush Baid, Jonathan Balloch, Patsorn Sangkloy, Vijay Upadhya, and Frank Dellaert. The dataset was obtained from Middlebury's stereo datasets, which can be found [here](http://vision.middlebury.edu/stereo/data/). Smoothing code was obtained from <https://github.com/beaupreda/semi-global-matching>.
