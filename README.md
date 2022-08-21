# FAsT-Match
FAsT-Match (Fast Affine Template Matching) is an algorithm designed by Simon Korman, Daniel Reichman, Gilad Tsur and Shai Avidan [(source)](https://www.cs.haifa.ac.il/~skorman/FastMatch/index.html) to search a fixed template inside an image, using the B&B technique.
This is a python implementation of the FAsT-Match algorithm with threshold learning option.

## The Algorithem
Branch-and-Bound (B&B) is a general technique for accelerating brute-force in large domains. It is used when a globally optimal is desirable, while more efficient optimization methods (like gradient descent) are irrelevant (e.g., due to the minimized function being nonconvex). A particular example of interest, very common in computer vision, is that of template matching (or image alignment), where one image (the template) is searched in another. This is useful, for example, in applications like stitching of panoramas, object localization and tracking.

The template which the FAsT-Match algorithm is trying to find, can be found scaled up/down, rotated and affined, therefore the number of transformations to consider is huge, and the higher bound on each sub-space of the domain is varied with respect to the current parameters and the loss function.

On each level of the algorithm, the alogorithem checks many transformations (also called configurations) which are spread along the search space. Each configuration is characterized by the distance between the template and the place on the image where this configuration transforms the template into, on sampled points. At the end of each level, we are left with an array of such distances, and we need to decide around which configurations to expand. Therefore, we use a threshold (the higher bound) on those distances’ values, compound of:

**threshold=minimum (called also best)  distance+f(δ)**

The following figure demonstrates how B&B works in FAsT-Match:

![tree_graphic](https://user-images.githubusercontent.com/87817221/185792647-5a916608-b29a-4fb4-9818-f9a3f7b0c74a.png)

The configurations in red were not close enough to expand in the next level (distance>threshold), while the configurations in blue, passed the threshold and therefore were chosen to be expanded in the next level of the algorithm (distance<threshold)

The f(δ) part of the bound is a linear combination of the delta hyper-parameter of the algorithm, which is responsible for the grid resolution of the search space, and it is lowered by the same factor each level. This linear combination has been computed manually through trial and error and it uses only one parameter – δ (delta), because of those reasons, it is not always optimal.

## Learning the Threshold
To tackle the previos issue, this project contains a Neural-Network model implemented in the FAsT-Match algorithm code and improves the decision process of the abovementioned higher bound, resulting in time and accuracy improvement of the FAsT-Match algorithm.

Now, we will consider the threashold to be:

**threshold=minimum distance+f(set of features)**

While the f(set of features) part is now taken from the Neural-Network model, and it isn’t only using a single parameter but a set of features (which includes delta).

The value which our model returns, ideally should be higher than the nearest configuration in the level, the one which on expansion will eventually get us to the real configuration of the template – the ground truth (green circle in the figure above).

The model used is a Multi-Layer Perceptron for Regression model, which is a simple Neural-Network consisting of some fully-connected linear layers.

### The final model configuration:

![model_scheme](https://user-images.githubusercontent.com/87817221/185792912-9efb99a2-dd8c-449e-acfa-2b0eb7be0d48.png)

The target value that was used is:
__min_distance+factor*(ground_truth-min_distance)__

By ranning on various target values to compare the results of each of them to the version of the algorithm without any such model the optimal factor can be chosen.

We compared the speed results by the average time per run, and the accuracy results by the Jaccard Index as well as the average distance between the corners of the found template to the ground truth one.

On the final round, we measured the results of 15 models with factor in the range 0.15-0.6 and got the following results on 400 test runs:
