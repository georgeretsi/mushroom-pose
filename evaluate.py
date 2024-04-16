import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import sys
import argparse
import copy

import imageio
from scipy.spatial.transform import Rotation
import time

sys.path.append('../scene_segmentation')
from remove_planes import remove_main_plane



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--pcd_file', action='store', type=str)
args = parser.parse_args()

pcd_file = args.pcd_file


from create_scene import scene_generation
from models.utils import process_input


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


name_prefix = 'nn'

from models.resunet import ResUNet3D

# model_path = 'ResUNetBN2C-32feat-3DMatch.pth'
# checkpoint = torch.load(model_path)
# backbone_model.load_state_dict(checkpoint['state_dict'])
model = ResUNet3D(1, 5)
model.load_state_dict(torch.load('./nn_model.pt'))
model = model.cuda()
#model.eval()



#pcd_file = '../synthetic_pcloud/synthetic_dataset/scene_15.ply'
#pcd_file = '../tmp_data/basic_scene/2.ply'
#pcd_file = '../tmp_data/cadcam_stereo/1.ply'
#pcd_file = '../tmp_data/cadcam_stereo/s1.ply'
pcd_file = '../tmp_data/reconstructed_pcds/3.ply'

# load real scene
#pcd_file = "./tmp_merged.ply"
#new_model.pt
#pcd_file = "./merged.ply"
#pcd_file = '../cadcam_stereo/exp_37-33_camera1.pcd'
scene_pcd = o3d.io.read_point_cloud(pcd_file)
#pp = np.asarray(scene_pcd.points)
#ids = np.where(np.linalg.norm(pp, axis=-1) < 0.75)[0]
#scene_pcd = scene_pcd.select_by_index(ids)


if 'basic_scene' in pcd_file:
    from scipy.spatial.transform import Rotation
    scene_pcd.rotate(Rotation.from_euler('zyx', [0.0 * np.pi, np.pi, 0.0 * np.pi]).as_matrix(), center=(0,0,0))


# open3d: find k nearest neighbors for each pcd point
# first build a tree structure
# then query the tree for each point
kdtree = o3d.geometry.KDTreeFlann(scene_pcd)
points = np.asarray(scene_pcd.points)
mdist = []
for i in range(1000):
    _, _, dists = kdtree.search_knn_vector_3d(points[np.random.randint(0, points.shape[0])], 11)
    mdist += [np.median(dists[1:])]
mdist = np.median(mdist)
print(mdist)
# find the nearest neighbors for each point


# scale to adapt to the voxel size!
mdist = np.median(np.asarray(scene_pcd.compute_nearest_neighbor_distance()))
print(mdist)
#scene_pcd.points = o3d.utility.Vector3dVector(.001 * np.asarray(scene_pcd.points) / mdist)

voxel_size = 0.005
#scene_pcd = scene_pcd.voxel_down_sample(voxel_size=.5 * voxel_size)
#scene_pcd, _, plane_model = remove_main_plane(scene_pcd, voxel_size=voxel_size)

scene_pcd.points = o3d.utility.Vector3dVector(1.0 * np.asarray(scene_pcd.points))

scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)

search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=30)
scene_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=30))
scene_pcd.orient_normals_to_align_with_direction(orientation_reference=np.asarray([0., 0., 1.]))
hfeatures = o3d.pipelines.registration.compute_fpfh_feature(scene_pcd, search_param).data.T

#scene_pcd.points = o3d.utility.Vector3dVector(ref_mdist * np.asarray(scene_pcd.points) / mdist)

pcd_center = scene_pcd.get_center()

#o3d.visualization.draw_geometries([scene_pcd])
#create_rotating_gif([scene_pcd], pcd_center, 'initial_pcd')

# ref!!!
#scene_pcd, _, _, _, _ = scene_generation(35, voxel_size=tvoxel_size)
#ref_mdist = np.asarray(ref_pcd.compute_nearest_neighbor_distance()).mean()


with torch.no_grad():
    tinput, point_inds, _ = process_input(xyz=np.asarray(scene_pcd.points),
                                          voxel_size=np.random.uniform(.9, 1.1) * voxel_size,
                                          device=device)


    encoded_features = model(tinput)
    reduced_points = np.asarray(scene_pcd.points)[point_inds]

    tcolors = np.asarray(scene_pcd.colors)[point_inds]

    pred_center = encoded_features[:, :3]
    pred_orient = encoded_features[:, -2].unsqueeze(-1)
    pred_class = encoded_features[:, -1].unsqueeze(-1)

svalues = pred_class.sigmoid()
vmask = svalues.squeeze() > .5
confs = pred_orient


npcd = o3d.geometry.PointCloud()
npcd.points = o3d.utility.Vector3dVector(reduced_points)
npcd.colors = o3d.utility.Vector3dVector(tcolors)

label_color = np.asarray([[c.cpu(), .0, .0] if b else [0, .8, 0] for c, b in zip(confs, vmask)])

npcd.colors = o3d.utility.Vector3dVector(label_color)

o3d.visualization.draw_geometries([npcd])




npcd = o3d.geometry.PointCloud()
npcd.points = o3d.utility.Vector3dVector(reduced_points - pred_center.cpu().numpy())
#npcd.points = o3d.utility.Vector3dVector((encoded_features[:, :3].data).cpu().numpy())
#npcd.points = o3d.utility.Vector3dVector((reduced_points).cpu().numpy())

#from sklearn.manifold import TSNE
#label_color = TSNE(n_components=3, learning_rate='auto',
#                  init='random', perplexity=3).fit_transform(encoded_features[:, :-1].cpu().data.numpy())

#from sklearn.decomposition import PCA
#label_color = PCA(n_components=3).fit_transform(reduced_features[:, :].cpu().data.numpy())

#label_color = encoded_features[:, :-1].cpu().data.numpy()
#label_color = (reduced_points - encoded_features[:, :3]).cpu().data.numpy()
#label_color = (label_color - label_color.min()) / (label_color.max() - label_color.min())

#label_color = np.asarray([[v.item(), 0, 0]  for v in svalues])
label_color = np.asarray([[1.0, 0, 0] if v else [.1, .1, .1] for v in vmask])
npcd.colors = o3d.utility.Vector3dVector(label_color)

o3d.visualization.draw_geometries([npcd])
#create_rotating_gif([npcd], pcd_center, 'center_pcd')


#'''
npcd = o3d.geometry.PointCloud()
#npcd.points = o3d.utility.Vector3dVector((reduced_points - encoded_features[:, :3].data).cpu().numpy())
#npcd.points = o3d.utility.Vector3dVector((encoded_features[:, :3].data).cpu().numpy())
npcd.points = o3d.utility.Vector3dVector(reduced_points)

#from sklearn.manifold import TSNE
#label_color = TSNE(n_components=3, learning_rate='auto',
#                  init='random', perplexity=3).fit_transform(encoded_features[:, :-1].cpu().data.numpy())

#from sklearn.decomposition import PCA
#label_color = PCA(n_components=3).fit_transform(reduced_features[:, :].cpu().data.numpy())

#label_color = encoded_features[:, :-1].cpu().data.numpy()
#label_color = (reduced_points - encoded_features[:, :3]).cpu().data.numpy()
#label_color = (label_color - label_color.min()) / (label_color.max() - label_color.min())

#label_color = np.asarray([[1.0, 0, 0] if v else [.1, .1, .1] for v in vmask])

label_color = np.asarray([[c.cpu(), .0, .0] if b else [0, .8, 0] for c, b in zip(confs, vmask)])

npcd.colors = o3d.utility.Vector3dVector(label_color)

o3d.visualization.draw_geometries([npcd])
#'''

#'''
preds = (reduced_points - pred_center).detach().cpu().numpy()[vmask.cpu()]
#preds = (reduced_points[:, :3]).detach().cpu().numpy()[vmask.cpu()]
#preds = (encoded_features[:, :-1]).detach().cpu().numpy()[vmask.cpu()]

from sklearn.cluster import MeanShift, DBSCAN
import time

tnow = time.time()
clusters = MeanShift(bandwidth=.01, min_bin_freq=2,  bin_seeding=True, cluster_all=True).fit(preds)
#clusters = MeanShift(bandwidth=.33, min_bin_freq=2,  bin_seeding=True, cluster_all=True).fit(preds)
print(time.time() - tnow)
#clusters = DBSCAN(eps=1.5 * voxel_size, min_samples=10).fit(reduced_points.detach().cpu().numpy()[vmask.cpu()])
print(clusters.labels_.max())

pclasses = -np.ones(reduced_points.size(0))
pclasses[vmask.cpu()] = clusters.labels_

from matplotlib.pyplot import cm
cmap = cm.get_cmap('tab20')

mpcds = []
for ii in range(-1, clusters.labels_.max()+1):
    tmask = pclasses == ii

    if sum(tmask) > 10:

        tpcd = o3d.geometry.PointCloud()
        tpcd.points = o3d.utility.Vector3dVector(reduced_points.cpu().numpy()[tmask])

        #  expand cluster and the mask to include all neighboring points within radius


        mdist = np.asarray(tpcd.compute_nearest_neighbor_distance()).mean()
        if mdist > 1.5 * voxel_size:
            continue

        #tpcd, _ = tpcd.remove_statistical_outlier(5, 2 * tvoxel_size)
        tpcd.paint_uniform_color(cmap(ii+1)[:3])

        mpcds += [(ii, tpcd)]
print(len(mpcds))
#o3d.visualization.draw_geometries(mpcds)

#create_rotating_gif(mpcds, pcd_center, 'clustered_pcd')


import matplotlib
cmap = matplotlib.cm.get_cmap('hsv')

'''
template_mesh = o3d.io.read_triangle_mesh('../cap_medium.ply')
template_mesh.vertices = o3d.utility.Vector3dVector(.0004 * np.asarray(template_mesh.vertices))
template_mesh.compute_vertex_normals()
template_pcd = template_mesh.sample_points_uniformly(10000)
#voxel_size = 0.0005
template_pcd_down = template_pcd.voxel_down_sample(voxel_size=0.005)
template_pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

template_points_inds, template_rfeatures, (template_red_points, template_red_feats) = extract_features(
    embed_model,
    xyz=1.0 * np.array(template_pcd_down.points),
    voxel_size=0.005,
    device=device,
    skip_check=True,
    reduced=True,
    is_eval=True)

template_fnl_features = models(template_red_feats.detach())
'''

#from create_scene import single_mushroom_feats

#template_points, template_feats, template_mesh = single_mushroom_feats(voxel_size=tvoxel_size)

#vmask = encoded_features[:, -1].sigmoid() > .5
#npcd.points = o3d.utility.Vector3dVector((reduced_points).cpu().numpy())

#label_color = np.asarray([[c.cpu(), .0, .0] if b else [0, .8, 0] for c, b in zip(confs, vmask)])

#npcd.colors = o3d.utility.Vector3dVector(label_color)


template_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
#template_mesh = template_mesh.sample_points_uniformly(500)
pp = np.asarray(template_mesh.vertices)
ids = np.where(pp[:, -1] > 0.01)[0]
template_mesh = template_mesh.select_by_index(ids)

from ellipsoid_align import ellipsoid_align

mpcds = mpcds[1:] #remove background
color_ids = np.linspace(0, 1, len(mpcds))
#vtmps = o3d.geometry.PointCloud()
vtmps = o3d.geometry.TriangleMesh()
#encoded_features = encoded_features.cpu().numpy()
encoded_features = torch.cat([pred_center, pred_orient, pred_class], dim=1).cpu().numpy()
reduced_points = reduced_points.cpu().numpy()
cnt = 0
for i, scene_part in mpcds:

    _, inds = scene_part.remove_statistical_outlier(5, 1.0)
    if len(inds) < 10:
        continue

    scene_part = scene_part.select_by_index(inds)

    # extend the region to include all points within radius
    rinds = np.asarray(npcd.compute_point_cloud_distance(scene_part)).squeeze() < 3 * voxel_size
    scene_part = npcd.select_by_index(rinds)

    scene_fnl_features = encoded_features[rinds]
    scene_fnl_points = reduced_points[rinds]

    #mcolor = np.median(np.asarray(scene_part.colors), axis=0)

    # check if color is too light
    #if (mcolor.mean() < .6) and (mcolor.max() < .8):
    #    continue

    '''
    # find mean color of all points in cluster
    mcolor = np.median(np.asarray(scene_part.colors)[inds], axis=0)

    # check if color is too light
    if (mcolor.mean() < .2) and (mcolor.max() < .7):
        continue

    scene_fnl_features = encoded_features[pclasses == i][inds]
    scene_fnl_points = reduced_points[pclasses == i][inds]
    '''



    R, t, s = ellipsoid_align(scene_fnl_points, scene_fnl_features)




    tmp_mesh = copy.deepcopy(template_mesh)
    tmp_mesh.vertices = o3d.utility.Vector3dVector(
        t + np.matmul(np.asarray(tmp_mesh.vertices) / s, np.linalg.pinv(R.T))
    )
    #tmp_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(tmp_mesh.vertices) / s)
    #tmp_mesh.rotate(np.linalg.pinv(R))
    #tmp_mesh.translate(t, relative=True)
    tmp_mesh.paint_uniform_color(cmap(color_ids[cnt])[:3])
    cnt += 1

    vtmps += tmp_mesh

lpcd = o3d.geometry.PointCloud()
lpcd.points = o3d.utility.Vector3dVector(reduced_points)
label_color = np.asarray([[1.0, 0, 0] if v else [.4, .4, .4] for v in vmask])
lpcd.colors = o3d.utility.Vector3dVector(label_color)

#o3d.visualization.draw_geometries([lpcd] + [vtmps])
o3d.visualization.draw_geometries([scene_pcd] + [vtmps])

#create_rotating_gif([scene_pcd] + [vtmps], pcd_center, 'final_pcd')
