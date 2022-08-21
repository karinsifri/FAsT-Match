from FastMatch import FastMatch
from CreateSamples import random_template
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.geometry import Polygon
import time
import MLP


def example_run(image, template, real_corners, model=None):
    fm = FastMatch()
    result_image = image.copy()
    corners, _, _ = fm.run(image, template, mlp_model=model)
    print("Actual corners:")
    print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])

    cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
    cv2.polylines(result_image, [corners], True, (255, 0, 0), 1)

    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.figure()
    plt.imshow(template, cmap='gray')
    plt.title("template")
    plt.figure()
    plt.imshow(result_image)
    plt.title("result")

    plt.show()


def corner_dist(corners1, corners2):
    corners1_copy = np.copy(corners1)
    corners1_copy = np.squeeze(corners1_copy)
    dist1 = corners1_copy - corners2
    dist1 = np.square(dist1)
    dist1 = np.sum(dist1, axis=1)
    dist1 = np.sqrt(dist1)
    dist1 = np.sum(dist1)

    corners1_copy[[1, 3]] = corners1_copy[[3, 1]]
    dist2 = corners1_copy - corners2
    dist2 = np.square(dist2)
    dist2 = np.sum(dist2, axis=1)
    dist2 = np.sqrt(dist2)
    dist2 = np.sum(dist2)

    return min([dist1, dist2])


def jaccard_index(pol1_xy, pol2_xy):
    pol1_xy_copy = np.copy(pol1_xy)
    pol1_xy_copy = np.squeeze(pol1_xy_copy)

    # Define each polygon
    polygon1_shape = Polygon(pol1_xy_copy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
    return polygon_intersection / polygon_union


def check_model(iterations, image, model):
    fm = FastMatch()
    corners_distance_with = []
    jaccard_index_with = []
    time_with = []
    corners_distance_without = []
    jaccard_index_without = []
    time_without = []
    result_image = image.copy()
    for i in range(iterations):
        template, real_corners = random_template(image)

        tic = time.time()
        corners_without, _, _ = fm.run(image, template)
        print("Actual corners:")
        print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
        time_without.append(time.time() - tic)
        corners_distance_without.append(corner_dist(corners_without, real_corners))
        jaccard_index_without.append(jaccard_index(corners_without, real_corners))

        tic = time.time()
        corners_with, _, _ = fm.run(image, template, mlp_model=model)
        print("Actual corners:")
        print(real_corners[0], real_corners[1], real_corners[2], real_corners[3])
        time_with.append(time.time() - tic)
        corners_distance_with.append(corner_dist(corners_with, real_corners))
        jaccard_index_with.append(jaccard_index(corners_with, real_corners))

        cv2.polylines(result_image, [real_corners], True, (0, 255, 0), 1)
        cv2.polylines(result_image, [corners_without], True, (255, 0, 0), 1)
        cv2.polylines(result_image, [corners_with], True, (138, 43, 226), 1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("image")
    plt.subplot(1, 2, 2)
    plt.imshow(result_image)
    plt.title("result")

    plt.figure()
    plt.scatter(range(1, iterations + 1), time_without, c="red")
    plt.scatter(range(1, iterations + 1), time_with, c="magenta")
    plt.legend(["without model", "with model"])
    plt.title("time")
    plt.figure()
    plt.scatter(range(1, iterations + 1), corners_distance_without, c="red")
    plt.scatter(range(1, iterations + 1), corners_distance_with, c="magenta")
    plt.legend(["without model", "with model"])
    plt.title("corners distance")
    print(corners_distance_with, time_with, corners_distance_without, time_without)
    plt.figure()
    plt.scatter(range(1, iterations + 1), jaccard_index_without, c="red")
    plt.scatter(range(1, iterations + 1), jaccard_index_with, c="magenta")
    plt.legend(["without model", "with model"])
    plt.title("jaccard index")
    print(jaccard_index_with, time_with, jaccard_index_without, time_without)
    plt.show()


if __name__ == '__main__':
    # There are two options to test the algorithm, uncomment the one you would like to run.

    images_folder = r"Images"  # Any image can be chosen
    models_path = r"PyTorch_models"  # Choose which model you would like to test from the 'PyTorch_models' folder

    # OPTION 1
    # Run the FAsT-Match algorithm on an example image with a given/random template
    '''
    ex_image = cv2.imread(images_folder + "/thai_food.jpg")
    ex_image = cv2.cvtColor(ex_image, cv2.COLOR_BGR2RGB)
    
    # Given template:
    ex_template = cv2.imread(...)
    ex_real_corners = ...
    
    # Random template:
    ex_template, ex_real_corners = random_template(ex_image)
    
    # With model
    mlp_model = MLP.load_model(models_path + "/mlp0.241577.pth")
    example_run(ex_image, ex_template, ex_real_corners, mlp_model)
    
    # Without model
    example_run(ex_image, ex_template, ex_real_corners)
    '''

    # OPTION 2
    # Compare the original FAsT-Match algorithm to the algorithm with the improvement model
    '''
    mlp_model = MLP.load_model(models_path + "/mlp0.241577.pth")
    img = cv2.imread(images_folder + "/zurich_object0024.view05.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    templates_amount = 3
    check_model(templates_amount, img, mlp_model)
    '''
