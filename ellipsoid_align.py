import numpy as np


def ellipsoid_align(points, centers, orients, confs, K=5):

    w = np.ones(points.shape[0])

    confs = 1 + 0 * confs.reshape(-1, 1)
    rot_enc = orients.reshape(-1, 1)

    for _ in range(K):

        ws = confs * w.reshape(-1, 1)

        e_center = ((points - centers) * ws).sum(0) / ws.sum()
        npoints = points - e_center

        #ws = np.sqrt(w.reshape(-1, 1))

        l = np.linalg.lstsq(ws * npoints, ws * rot_enc, rcond=None)[0].reshape(-1)

        sz = np.linalg.norm(l)

        siny = -l[0] / sz
        cosy = np.sqrt(l[1] ** 2 + l[2] ** 2) / sz

        sinx = (l[1] / sz) / cosy  # (l[1] / l[2]) / np.sqrt((l[1] / l[2]) ** 2 + 1)
        cosx = np.sqrt(1 - sinx ** 2)  # 1 / np.sqrt((l[1] / l[2]) ** 2 + 1)

        e_rot = np.asarray([
            [cosy, sinx * siny, cosx * siny],
            [0, cosx, -sinx],
            [-siny, sinx * cosy, cosx * cosy]
        ])
        tpoints = np.matmul(npoints, e_rot.T)

        predefined_s = np.asarray([1, 1, 1.5])

        #s = min(sz / 1.5, np.sqrt(np.median(1.0 / ((tpoints ** 2) @ (predefined_s ** 2).reshape(3, 1)))))
        #s =  np.sqrt(np.mean(1.0 / ((tpoints ** 2) @ (predefined_s ** 2).reshape(3, 1))))
        #s = min(sz / 1.5, np.sqrt((ws / ((tpoints ** 2) @ (predefined_s ** 2).reshape(3, 1))).sum()/ws.sum()))
        s = max(sz / 1.5, np.sqrt((ws / ((tpoints ** 2) @ (predefined_s ** 2).reshape(3, 1))).sum() / ws.sum()))

        # s = sz / 1.5

        e_scale = s * predefined_s

        w = 1 / np.maximum(1e-5,  np.abs(1 - (s**2) * (tpoints ** 2) @ (predefined_s ** 2).reshape(3, 1)))

    return e_rot, e_center, e_scale


import numpy as np
from scipy.spatial import KDTree


    # Initialize transformation matrix (translation, rotation, and scale)
#if init_values is not None:
#    R, t, s = init_values['R'], init_values['t'], init_values['s']
#    target = t + np.matmul(target / s, np.linalg.pinv(R.T))


# icp R, t, s transfrom based on correspondences
def icp_transform(src_points, tgt_points,):

    # use Umeyama algorithm to compute transformation

    # compute centroids
    src_centroid = np.mean(src_points, axis=0)
    tgt_centroid = np.mean(tgt_points, axis=0)

    # compute covariance matrix
    src_cov = np.matmul((src_points - src_centroid).T, (src_points - src_centroid))
    tgt_cov = np.matmul((tgt_points - tgt_centroid).T, (tgt_points - tgt_centroid))

    # compute SVD
    U, _, V = np.linalg.svd(np.matmul(src_cov, tgt_cov))

    # compute rotation matrix
    R = np.matmul(U, V)

    # compute translation vector
    t = tgt_centroid - np.matmul(R, src_centroid)

    # compute scale
    s = np.trace(np.matmul(R, src_cov)) / np.trace(src_cov)

    # compute transformation matrix (translation, rotation, and scale)
    T = np.eye(4)
    # add scale and rotation
    T[:3, :3] = s * R
    # add translation
    T[:3, 3] = t

    # apply transformation to source points
    transformed = np.matmul(src_points, T[:3, :3].T) + T[:3, 3]

    return R, t, s


from scipy.spatial import procrustes


def find_rigid_transform(src_points, dst_points):

    # Check if the number of points in both sets match
    if src_points.shape != dst_points.shape:
        raise ValueError("Input point sets must have the same shape.")

    # Compute centroids of both point sets
    src_centroid = np.mean(src_points, axis=0)
    dst_centroid = np.mean(dst_points, axis=0)

    # Center the points by subtracting the centroids
    src_centered = src_points - src_centroid
    dst_centered = dst_points - dst_centroid

    # Compute the cross-covariance matrix
    cross_cov_matrix = np.dot(src_centered.T, dst_centered)

    # Compute the SVD of the cross-covariance matrix
    U, _, VT = np.linalg.svd(cross_cov_matrix)

    # Compute the optimal rotation matrix
    rotation_matrix = np.dot(VT.T, U.T)

    # Compute the optimal scale
    scale = np.linalg.norm(dst_centered) / np.linalg.norm(src_centered)

    # Compute the optimal translation vector
    translation_vector = dst_centroid - scale * np.dot(src_centroid, rotation_matrix.T)

    return rotation_matrix, translation_vector, scale


def icp_registration(src_points, tgt_points, src_features, tgt_features, init_transform, voxel_size, max_iters=5, l=10, prune=True):

    src_features = src_features.reshape(-1, 1)
    tgt_features = tgt_features.reshape(-1, 1)

    # add features to source and target points as extra column
    src_points_h = np.hstack((src_points, l * voxel_size * src_features))

    # create KD tree for target points
    tree = KDTree(src_points_h)


    # Initialize transformation matrix (translation, rotation, and scale)
    if init_transform is not None:
        R, t, s = init_transform['R'], init_transform['t'], init_transform['s']
        # change transformation matrices to be in line with this implementation
        s = 1/s
        R = np.linalg.pinv(R)
        ntgt_points = np.matmul(tgt_points, (s * R).T) + t
    else:
        R, t, s = np.eye(3), np.zeros(3), 1.0
        ntgt_points = tgt_points


    for i in range(max_iters):

        tgt_points_h = np.hstack((ntgt_points, l * voxel_size * tgt_features))
        # find unique correspondences between source and target points
        dist, idx = tree.query(tgt_points_h)
        idx, tgt_idx = np.unique(idx, return_index=True)

        # prune correspondences based on distance
        dist = dist[idx]
        dist_mask = dist < 1.1 * np.median(dist)
        #dist_mask = dist < (.5 * l + 1.5) * voxel_size
        idx = idx[dist_mask]
        tgt_idx = tgt_idx[dist_mask]


        if prune:
            eidx = (np.abs(tgt_features[tgt_idx] - src_features[idx]) < .2).reshape(-1)
            if sum(eidx) > 3:
                idx = idx[eidx]
                tgt_idx = tgt_idx[eidx]
            else:
                break

        # compute transformation based on correspondences
        # src_features[idx], target_features[tgt_idx],
        #R, t, s = icp_transform(tgt_points[tgt_idx], src_points[idx])
        R, t, s = find_rigid_transform(tgt_points[tgt_idx], src_points[idx])

        # update source points
        ntgt_points = np.matmul(tgt_points, (s * R).T) + t

    # change again transformation matrices to match initial convention
    s = 1/s
    R = np.linalg.pinv(R)

    return R, t, s

