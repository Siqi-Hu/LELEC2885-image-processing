import numpy as np

import imageio
# Load the image
imfile = imageio.imread('images/peppers.png')
image  = np.array(imfile)/255 # Normalize between 0 and 1

# Reshape the image as a list of pixels
X = image.reshape((image.shape[0]*image.shape[1],3))


## TO IMPLEMENT : K-means algorithm

def kmeans(X, K, initype='random', maxIter=100, plotIterations=True):
    """Some example here for good practice"""
    (N, dim) = X.shape
    # Initialization

    """ Switch intialization. Don't forget to reset N,X and K by running the upper cell """
    if (initype == 'random'):
        centroids = np.random.randn(K, dim)  # Random init
    elif (initype == 'kmeans_pp'):
        centroids = kmeans_pp_init(X, K)
    assignments = np.zeros(N)
    iterate = 0

    # TO COMPLETE :-)
    while iterate < maxIter:
        print("iter:", iterate)

        for i in range(N):
            assignments[i] = np.argmin(np.linalg.norm(X[i] - centroids, axis=1))

        # check if there exist some dead centroids
        # when there is a dead centroid, reassign it to the other point in the other groups
        group_sizes = [X[assignments == i].shape[0] for i in range(K)]
        while (0 in group_sizes) or (1 in group_sizes):
            for i in range(K):
                # if group_sizes[i] == 0:  # if the centroids has no point in its group, reassign this centroid with a random point
                #     random_point = np.random.randn(dim)
                #     while random_point in centroids:
                #         random_point = np.random.randn(dim)
                #     centroids[i] = random_point


                if group_sizes[i] <= 1:  # if the centroids has no point in its group, reassign this centroid with a random point
                    random_point = X[np.random.randint(N), :]
                    while random_point in centroids:
                        random_point = X[np.random.randint(N), :]
                    centroids[i] = random_point

            for i in range(N):
                assignments[i] = np.argmin(np.linalg.norm(X[i] - centroids, axis=1))

            group_sizes = [X[assignments == i].shape[0] for i in range(K)]

        # step 2: updating centroids
        print("group sizes: ", group_sizes)
        for i in range(K):
            group = X[assignments == i]
            print(group)
            print(group.shape)

            # total_dist_to_c = np.zeros(group.shape[0])
            # for j in range(group.shape[0]):
            #     c = group[j]
            #     total_dist_to_c[j] = np.linalg.norm(group - c, axis=1).sum()



            mean_point = np.mean(group - centroids[i], axis=0)
            centroids[i, :] = group[np.argmin(np.linalg.norm(group - mean_point, axis=1)), :]


        iterate += 1

    print('K-means terminated after {} iterations'.format(iterate))

    return centroids, assignments


# TO COMPLETE : write the code implementing the kmeans++ initialization algorithm

def kmeans_pp_init(X, K):
    (N, dim) = X.shape
    # Initialization
    centroids = np.zeros((K, dim))  # TO BE MODIFIED

    for k in range(K - 1):

        if k == 0:
            idx = int(np.random.uniform(N))
            centroids[k, :] = X[idx, :]
            continue

        current_centroids = centroids[:k, :]
        dist_square_dict = dict()
        for i in range(N):
            x = X[i, :]
            dist = np.min(np.linalg.norm(x - current_centroids, axis=1))
            dist_square = np.square(dist)
            dist_square_dict[i] = dist_square

        sum_dist_square = sum(dist_square_dict.values())
        prob_dict = {id: dist / sum_dist_square for id, dist in dist_square_dict.items()}

        prob_dict_sorted = dict(sorted(prob_dict.items(), key=lambda x : x[1], reverse=True))
        random_choice = np.random.random()

        new_centroid_id = None
        cum = 0
        for idx, prob in prob_dict_sorted.items():
            if (cum < random_choice) and (cum + prob > random_choice):
                new_centroid_id = idx
                break
            else:
                cum += prob

        centroids[k, :] = X[new_centroid_id, :]


    return centroids

# # TO RUN
# # with different values of K
# K = 5
# maxIter = 3 # Careful, increasing this might make the algorithm slow
# (centroids,assignments) = kmeans(X,K,initype='random',maxIter = maxIter,plotIterations = False)
#

#%%
# TO RUN
N   = 100
dim = 2
X   = 0.45*np.random.randn(N,dim) + 2.75*np.random.randint(2,size=(N,dim))-1
K   = 4
(c,assignments) = kmeans(X,K, initype = 'kmeans_pp')

print(c)