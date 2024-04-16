import  numpy as np
import torch
import open3d as o3d

# visualization func
import matplotlib

def visualize_features(feats, thres=.5, mode='center'):

    # mode can be either 'center' or 'values'
    # check if mode is valid
    assert mode in ['center', 'values'], 'mode can be either "center" or "values"'

    npcd = o3d.geometry.PointCloud()
    reduced_points = feats['points']
    if mode == 'center':
        pred_center = feats['pred_center']
        npcd.points = o3d.utility.Vector3dVector(reduced_points - pred_center)
    else:
        npcd.points = o3d.utility.Vector3dVector(reduced_points)

    pred_orient = feats['pred_orient']
    vmask = feats['pred_class'] > thres
    tcolors = np.asarray([[c, .0, .0] if b else [.2, .8, .2] for c, b in zip(pred_orient, vmask)])
    npcd.colors = o3d.utility.Vector3dVector(tcolors)

    o3d.visualization.draw_geometries([npcd])

def visualize_clusters(feats, pclasses):
    cmap = matplotlib.cm.get_cmap('tab20')

    points = feats['points']
    orient = feats['pred_orient']

    mpcds = []
    for ii in range(-1, int(pclasses.max()) + 1):
        tmask = pclasses == ii

        if sum(tmask) > 10:

            tpcd = o3d.geometry.PointCloud()
            tpcd.points = o3d.utility.Vector3dVector(points[tmask])


            mdist = np.asarray(tpcd.compute_nearest_neighbor_distance()).mean()
            if mdist > 1.5 * voxel_size:
                continue

            # tpcd, _ = tpcd.remove_statistical_outlier(5, 2 * tvoxel_size)
            #tpcd.paint_uniform_color(cmap(ii + 1)[:3])
            tc = cmap(ii + 1)[:3]
            if ii >= 0:
                tcolors = np.asarray([o * tc for o in orient[tmask]])
                tpcd.colors = o3d.utility.Vector3dVector(tcolors)
            else:
                tpcd.paint_uniform_color(tc)

            mpcds += [(ii, tpcd)]

    o3d.visualization.draw_geometries([tpcd for _, tpcd in mpcds])

# plane segmentation

def remove_main_plane(pcd, voxel_size):

    #pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
    clear_pcd = copy.deepcopy(pcd)
    plane_model, inliers = clear_pcd.segment_plane(distance_threshold=3 * voxel_size,
                                             ransac_n=3,
                                             num_iterations=1000)

    tplane = clear_pcd.select_by_index(inliers)
    clear_pcd = clear_pcd.select_by_index(inliers, invert=True)

    return clear_pcd, tplane, plane_model

# model definition and loading
def load_model(load_path, device):
    from models.resunet import ResUNet3D

    model = ResUNet3D(1, 5)
    model.load_state_dict(torch.load(load_path))
    model = model.to(device)
    model.eval()

    return model

from models.utils import process_input

# use model to get features
def get_features(model, scene_pcd, voxel_size, device):
    points = np.asarray(scene_pcd.points)
    tinput, point_inds, _ = process_input(xyz=points,
                                          voxel_size=voxel_size,
                                          device=device)

    with torch.no_grad():
        encoded_features = model(tinput)


    #reduced_points = np.asarray(scene_pcd.points)[point_inds]
    #tcolors = np.asarray(scene_pcd.colors)[point_inds]

    pred_center = encoded_features[:, :3].cpu().numpy()
    pred_orient = encoded_features[:, -1].unsqueeze(-1).cpu().numpy()
    pred_class = encoded_features[:, -2].sigmoid().unsqueeze(-1).cpu().numpy()

    feats = {
        'points': points[point_inds],
        'colors': np.asarray(scene_pcd.colors)[point_inds],
        'pred_center': pred_center,
        'pred_orient': pred_orient,
        'pred_class': pred_class,
    }

    return feats, point_inds


from sklearn.cluster import MeanShift

# clustering
def segment_clusters(feats, voxel_size, thres=.5, soft=False):

    vmask = (feats['pred_class'] > thres).squeeze()
    preds = (feats['points'] - feats['pred_center'])

    if soft:
        pclass = feats['pred_class']
        preds = np.column_stack((preds, voxel_size * pclass))

    clusters = MeanShift(bandwidth=3 * voxel_size, min_bin_freq=10, bin_seeding=True, cluster_all=True).fit(preds[vmask])

    # cluster centers
    centers = clusters.cluster_centers_

    # find the distance to the closest cluster center
    dists = np.linalg.norm(preds[:, None] - centers[None], axis=-1)

    # assign each point to the closest cluster center if the distance is less than 1.5 * voxel_size
    pclasses = np.argmin(dists, axis=-1)
    pclasses[dists[np.arange(dists.shape[0]), pclasses] > 2.5 * voxel_size] = -1

    # find if clusters.Labels_ has less than 10 points


    #pp = clusters.predict(feats['points'] - feats['pred_center'])


    #pclasses = -np.ones(vmask.shape[0])
    #pclasses[vmask] = clusters.labels_

    #pclasses =  clusters.predict(feats['points'] - feats['pred_center'])

    return pclasses


from ellipsoid_align import ellipsoid_align
import copy

def visualize_ellipsoids(estimates, scene_pcd):

    template_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    pp = np.asarray(template_mesh.vertices)
    ids = np.where(pp[:, -1] > 0.01)[0]
    template_mesh = template_mesh.select_by_index(ids)

    cmap = matplotlib.cm.get_cmap('hsv')
    color_ids = np.linspace(0, 1, len(estimates))

    vtmps = o3d.geometry.TriangleMesh()
    for cnt, (i, est) in enumerate(estimates):

        R, t, s = est['R'], est['t'], est['s']

        tmp_mesh = copy.deepcopy(template_mesh)
        tmp_mesh.vertices = o3d.utility.Vector3dVector(
            t + np.matmul(np.asarray(tmp_mesh.vertices) / s, np.linalg.pinv(R.T))
        )

        tmp_mesh.paint_uniform_color(cmap(color_ids[cnt])[:3])

        vtmps += tmp_mesh

    o3d.visualization.draw_geometries([scene_pcd, vtmps])

# pose estimation
def ellipsoid_pose_estimation(feats, pclasses, voxel_size):

    estimations = []
    for i in range(0, int(pclasses.max()) + 1):

        if  sum(pclasses == i) == 0:
            continue

        tpoints = feats['points'][pclasses == i]
        # find the closest neighbor of each point using numpy
        dists = np.linalg.norm(tpoints[:, None] - tpoints[None], axis=-1)
        dists += np.eye(dists.shape[0]) * 1e10
        cinds = np.argmin(dists, axis=-1)

        # find if the closest distance is less than 1.5 * voxel_size
        inds = np.where(dists[np.arange(dists.shape[0]), cinds] < 1.5 * voxel_size)[0]

        if len(inds) < 10:
            continue

        R, t, s = ellipsoid_align(feats['points'][pclasses == i][inds],
                                  feats['pred_center'][pclasses == i][inds],
                                  feats['pred_orient'][pclasses == i][inds],
                                  feats['pred_class'][pclasses == i][inds],
                                  K=1)

        est = {'R': R, 't': t, 's': s}
        estimations += [(i, est)]

    return estimations


def get_ellipsoid_template():
    template_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    pp = np.asarray(template_mesh.vertices)
    ids = np.where(pp[:, -1] > 0.01)[0]
    template_mesh = template_mesh.select_by_index(ids)
    mushroom_pcd = template_mesh.sample_points_uniformly(10000)

    mushroom_pcd = mushroom_pcd.voxel_down_sample(voxel_size=voxel_size)

    return mushroom_pcd

def get_mushroom_template():
    mushroom_filename = "./templates/mushrooms/mushroom_basic.obj"
    mushroom_mesh = o3d.io.read_triangle_mesh(mushroom_filename, True)
    mushroom_mesh = mushroom_mesh.remove_unreferenced_vertices()
    mushroom_mesh = mushroom_mesh.remove_duplicated_vertices()
    mushroom_mesh.compute_vertex_normals()
    mushroom_mesh.scale(0.005, center=mushroom_mesh.get_center())
    mushroom_mesh.translate((0, 0, 0.3), relative=False)
    mushroom_mesh.scale(0.1, center=mushroom_mesh.get_center())
    mushroom_pcd = mushroom_mesh.sample_points_uniformly(10000)

    mushroom_pcd = mushroom_pcd.voxel_down_sample(voxel_size=voxel_size)

    return mushroom_pcd


from ellipsoid_align import icp_registration
# using template alignment
def finetune_pose_estimation(scene_feats, template_feats, pclasses, init_estimates, voxel_size):

    # mushroom template load and preprocess

    estimations = []
    for i in range(0, int(pclasses.max()) + 1):

        if  sum(pclasses == i) == 0:
            continue

        tpoints = scene_feats['points'][pclasses == i]

        # find the closest neighbor of each point using numpy
        dists = np.linalg.norm(tpoints[:, None] - tpoints[None], axis=-1)
        dists += np.eye(dists.shape[0]) * 1e10
        cinds = np.argmin(dists, axis=-1)

        # find if the closest distance is less than 1.5 * voxel_size
        inds = np.where(dists[np.arange(dists.shape[0]), cinds] < 1.5 * voxel_size)[0]

        if len(inds) < 10:
            continue

        # find the element from list init_estimates (i, estimate) that has the same i
        if init_estimates is not None:
            for (ii, estimate) in init_estimates:
                if ii == i:
                    init_estimate = estimate
                    break
        else:
            init_estimate = None

        #init_estimate = init_estimates[i][1] if init_estimates is not None else None
        R, t, s = icp_registration(
            scene_feats['points'][pclasses == i][inds],
            template_feats['points'],
            scene_feats['pred_orient'][pclasses == i][inds],
            template_feats['pred_orient'],
            init_estimate,
            voxel_size
            )

        est = {'R': R, 't': t, 's': s}
        estimations += [(i, est)]

    return estimations

def color_postprocessing(scene_feats, pclasses, voxel_size):

    colors = scene_feats['colors']
    for i in range(0, int(pclasses.max()) + 1):

        if  sum(pclasses == i) == 0:
            continue

        tpoints = scene_feats['points'][pclasses == i]

        # find the closest neighbor of each point using numpy
        dists = np.linalg.norm(tpoints[:, None] - tpoints[None], axis=-1)
        dists += np.eye(dists.shape[0]) * 1e10
        cinds = np.argmin(dists, axis=-1)

        # find if the closest distance is less than 1.5 * voxel_size
        inds = np.where(dists[np.arange(dists.shape[0]), cinds] < 1.5 * voxel_size)[0]

        if len(inds) < 10:
            continue


        # color of extracted region
        tcolors = colors[pclasses == i][inds]

        # median color of extracted region
        mcolor = np.mean(tcolors @ np.asarray([0.299, 0.587, 0.114]).T)


        if mcolor < .7:
            pclasses[pclasses == i] = -1

        #print(mcolor)
    return pclasses

# main pipeline function
def run_pipeline(pcd_file, voxel_size, cthres=.5, device='cuda:0', visualize=False, plane_removal=True, color_preprocessing=True):

    # load pointcloud and downsample it
    if  'real_mushrooms_pcds' in pcd_file:
        tpcd = o3d.io.read_point_cloud(pcd_file)
        pp = np.asarray(tpcd.points)
        ids = np.where(np.linalg.norm(pp, axis=-1) < 0.9)[0]
        tpcd = tpcd.select_by_index(ids)
        tpcd = tpcd.voxel_down_sample(voxel_size)

        from scipy.spatial.transform import Rotation
        r = Rotation.from_euler('zyx', [0.0 * np.pi, np.pi, 0.0 * np.pi]).as_matrix()
        tpcd.rotate(r, center=(0,0,0))
        #tpcd = remove_unconsistent_points(tpcd)
        scene_pcd = tpcd.remove_statistical_outlier(15, 2.0)[0]
    else:
        scene_pcd = o3d.io.read_point_cloud(pcd_file)
        scene_pcd.points = o3d.utility.Vector3dVector(1.0 * np.asarray(scene_pcd.points))
        scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)

    if plane_removal:
        scene_pcd_cleared, _, _ = remove_main_plane(scene_pcd, voxel_size)
    else:
        scene_pcd_cleared = scene_pcd

    if color_preprocessing:
        colors = np.asarray(scene_pcd_cleared.colors)
        colors = colors @ np.asarray([0.299, 0.587, 0.114]).T
        cinds = np.where(colors > .3)[0]
        scene_pcd_cleared = scene_pcd_cleared.select_by_index(cinds)


    # load model
    #model = load_model('nn_model.pt', device)
    #model = load_model('fnl_model.pt', device)
    model = load_model('nn_model.pt', device)

    feats, point_inds = get_features(model, scene_pcd_cleared, voxel_size, device)

    if visualize:
        o3d.visualization.draw_geometries([scene_pcd])
        o3d.visualization.draw_geometries([scene_pcd_cleared])
        visualize_features(feats, thres=cthres, mode='values')
        visualize_features(feats, thres=cthres, mode='center')

    pclasses = segment_clusters(feats, voxel_size, thres=cthres)

    if visualize:
        visualize_clusters(feats, pclasses)

    pclasses = color_postprocessing(feats, pclasses, voxel_size)

    if visualize:
        visualize_clusters(feats, pclasses)

    pose_estimations = ellipsoid_pose_estimation(feats, pclasses, voxel_size)

    if visualize:
        visualize_ellipsoids(pose_estimations, scene_pcd)

    mushroom_pcd = get_ellipsoid_template() #get_mushroom_template()
    #mushroom_pcd.scale(1/pose_estimations[0][1]['s'].mean(), center=mushroom_pcd.get_center())
    #mfeats, _ = get_features(model, mushroom_pcd, voxel_size, device)

    pp = np.asarray(mushroom_pcd.points)
    mfeats = {
        'points': pp,
        'pred_orient': (pp[:, -1] - pp[:, -1].min()) / (pp[:, -1].max() - pp[:, -1].min())
    }

    pose_estimations = finetune_pose_estimation(feats, mfeats, pclasses, pose_estimations, voxel_size)

    if visualize:
        visualize_ellipsoids(pose_estimations, scene_pcd)

    return pose_estimations







# main func that load pcd and run the pipeline
if __name__ == '__main__':

    #pcd_file = './reconstructed_pcds/3.ply'
    #pcd_file = '../tmp_data/cadcam_stereo/s1.ply'

    pcd_file = './real_mushrooms_pcds/reconstruction_pcd_25.pcd'

    voxel_size = 0.004
    run_pipeline(pcd_file, voxel_size, cthres=.5, visualize=True, plane_removal=False)

