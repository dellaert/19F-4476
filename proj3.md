---
layout: default
title: Project 3 - Camera Projection Matrix and Fundamental Matrix Estimation with RANSAC
---

# Project 3 - Camera Projection Matrix and Fundamental Matrix Estimation with RANSAC

## Brief
* Due:
  * 10/16/2019 11:59PM - Parts 1, 2, and 3
* Project materials including writeup template [proj3_0.zip](projects/proj3_0.zip)
* Hand-in: through [Canvas](https://gatech.instructure.com) AND [Gradescope](https://www.gradescope.com)
* Required files:
  * `<your_gt_username>.zip` on Canvas
  * `<your_gt_username>_proj2.pdf` on Gradescope

## Setup
Note that the same environment used in projects 1 and 2 can be used for this project!!! If you already have a working environment, just activate it and you are all set, no need to redo these steps! If you run into import module errors, try "pip install -e ." again, and if that still doesn't work, you may have to create a fresh environment.

1. Install [Miniconda](https://conda.io/miniconda.html). It doesn't matter whether you use Python 2 or 3 because we will create our own environment that uses 3 anyways.
2. Create a conda environment using the appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Modify the command based on your OS (`linux`, `mac`, or `win`): `conda env create -f proj2_env_<OS>.yml`
3. This should create an environment named 'proj2'. Activate it using the Windows command, `activate proj2` or the MacOS / Linux command, `source activate proj2`
4. Install the project package, by running `pip install -e .` inside the repo folder.
5. Run the notebook using `jupyter notebook ./proj2_code/proj2.ipynb`
6. Ensure that all sanity checks are passing by running `pytest unit_tests` inside the repo folder.
7. Generate the zip folder for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>` and submit to Canvas (don't forget to submit your report to Gradescope!).

# Part 1 -- Projection Matrix
**Learning objective** Camera Calibration, using fiducial markers for camera calibration.

Estimating the geometry of a 3D scene, e.g. the position of the camera relative to a known object, can be done if the intrinsic and the extrinsic camera parameters are known. However, this is limited since from a single view scene structure and depth are inherently ambiguous. The importance of knowing camera parameters will become more clear in part 2 where you will use multiple views. Recall that for a pinhole camera model, the camera matrix $P \in \mathbb{R}^{3\times4}$ is a projective mapping from world (3D) to pixel (2D) coordinates defined up to a scale
{% raw %}
\mathbf{x} = \mathbf{P}\mathbf{X} = 
\begin{bmatrix}
    u \\
    v \\
    1
\end{bmatrix}
\cong
\begin{bmatrix}
    s \cdot u \\
    s \cdot v \\
    s
\end{bmatrix}
=
\begin{bmatrix}
m_{11} & m_{12} & m_{13} & m_{14} \\
m_{21} & m_{22} & m_{23} & m_{24} \\
m_{31} & m_{32} & m_{33} & m_{34} \\
\end{bmatrix}
\begin{bmatrix}
    x_w \\
    y_w \\ 
    z_w \\
    1
\end{bmatrix}.
{% endraw %}

## Part II: Fundamental Matrix
The main goal here is to formulate this problem as an optimization problem that tries to minimize an objective function defined as
$$J(F) = ind(Fp,q)2+ind(p,Fq)2$$
where n represents the total number of points
d represents the distance function for a line and a point
F is the fundamental matrix we want to optimize for
p is one point pair
q is another point pair
We also set the following constraints on F
3x3 matrix
Must be rank 2
Do we need to enforce this constraint?
Defined up to a scale
Frobenius norm is 1
Instructions (SEE Below for more details)
Code the objective function
Setup SciPy for optimization
Set up optimizer and constraints
Run optimization and get an output for F
Apply F yourself using own pictures
Coordinate normalization of F for better results
Fundamental Matrix song

Calculating Fundamental Matrix

Now that we know how to project a point from a 3D coordinate to a 2D coordinate, next we’ll look at how to map corresponding 2D points from two images of the same scene. In this part, given a set of corresponding 2D points, we will do calculations to estimate the fundamental matrix.

Note: the fundamental matrix is sometimes defined as the transpose of the above matrix with the left and right image points swapped. Both are valid fundamental matrices, but we assume you use the one above.


https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwiU2vzK2-fkAhXtguAKHSyBDUsQjRx6BAgBEAQ&url=https%3A%2F%2Fslideplayer.com%2Fslide%2F5279455%2F&psig=AOvVaw2QZO15Ap6vXFMJPvnjtGZg&ust=1569354563277107

You can think of the fundamental matrix as a transformation that takes points in one image and maps them to a line in another image. The reason why we want to solve for F such that the equation above is 0 is because ideally we want to end up with two orthogonal lines. We want the vector represented by [u, v, 1]^T to be orthogonal to [u’, v’, 1] multiplied by F. In other words, we want the fundamental matrix to transform our points [u’, v’, 1] to a line that is perpendicular to the line that includes q. This is also like saying that we want the line, [u’, v’, 1] multiplied by F, to be as close to point [u, v, 1].

As such, in order for the two lines to be perpendicular, we want to minimize the error of our mapping from point p to the epipolar line that is perpendicular to and intersects the line with point q.

We will give you a set of n points, and with each point, you can set up the equation
d(Fp,q)2+d(p,Fq)2
where d represents the distance, or the error, between line Fp and point q, and we add that to the error between line Fq and point p. We can do this for n pairs of points and create an objective function representing the sum of squares of these errors across all the points
J(F) = ind(Fp,q)2+ind(p,Fq)2
Now in order to calculate the d distance between a line and a point, we can do this by using projections


https://en.wikibooks.org/wiki/Linear_Algebra/Orthogonal_Projection_Onto_a_Line
And the value for d that we want is the magnitude of vector v - c_ps. v and s represent the projected line and point respectively, these are the Fp and q.

Now that we have an optimization function, we can use SciPy to run it through an optimizer and get our values of F. But before that, we also have to find a way to parameterize F (find a way to represent F with variables), so our optimizer knows what it needs to find.

The fundamental matrix is a 3x3 matrix, meaning there are 9 variables. However, since it is defined up to a scale we can take away 1 variable by setting the bottom right element of the matrix to 1. This gives us 8 degrees of freedom. If we also want to enforce that our matrix has rank 2, then we can go down to 7 degrees of freedom. In order to do that, we can parameterize our matrix in the following way.
 
Apply Fundamental Matrix Yourself
Now we test your implementation of how to find the fundamental matrix by using images and point pairs that you capture on your own.
Take picture, get points, optimize for fundamental matrix, test to see if values are correct
Coordinate normalization??
Fundamental Matrix Song


## Part III: RANSAC
Now you have a function which can calculate the Fundamental Matrix from matching pairs of points in two different images. However, having to manually extract the matching points is undesirable. In the previous project, we implemented SIFTNet to automate the process of identifying matching points in two images.

We will implement a pipeline to run two images through SIFTNet to extract matching points and then send those points to your fundamental matrix estimation function to acheive an automated process for generating the fundamental matrix. However, there is an issue with this. Previously, the manually identified points were perfect matches between your two images. However, as we saw before, SIFT does not generate matches with 100% accuracy. Fortunately, to calculate the fundamental matrix we need only 8 matching points and SIFT can generate hundreds, so there should be more than enough good matches somewhere in there, if only we can find them.

We will use a method called RANdom SAmple Consensus (RANSAC) to search through the points returned by SIFT and find true matches to use for calculating the fundamental matrix. You can find a simple explanation of RANSAC at https://www.mathworks.com/discovery/ransac.html See section 6.1.4 in the textbook for a more thorough explanation of how RANSAC works.

In summary, we will implement a workflow using the SIFTNet from project 2 to extract feature points, then RANSAC will select a random subset of those points, you will call your function from part 2 to calculate the fundamental matrix for those points, and then you will check how many other points identified by SIFTNet match this fundamental matrix. Then you will iterate through RANSAC again until you find the subset of points that produces the best fundamental matrix with the most matching points.
