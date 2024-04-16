import open3d as o3d
import os
import copy
import random
import numpy as np

#from scipy.interpolate import RBFInterpolator

def augment_template(template_pcd, valid_ids=None, voxel_size=0.001):

    points = np.asarray(template_pcd.points)


    scales = np.random.uniform(.99, 1.01) * np.asarray([
        np.random.uniform(.95, 1.05),
        np.random.uniform(.95, 1.05),
        np.random.uniform(.95, 1.05)
    ])

    points *= scales.reshape(1, 3)

    #points = scales.reshape(1, 3) * np.matmul(points, R) + T.reshape(1, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)



    #'''
    # add noise !!
    N = len(pcd.points)
    K = int(.1 * N)

    target_ids = np.zeros(N)

    # remove few random points
    ridxs = np.random.choice(len(pcd.points), K, replace=False)

    if valid_ids is not None:
        target_ids[valid_ids] = 1
    else:
        target_ids[:] = 1
    target_ids = np.delete(target_ids, ridxs)

    pcd = pcd.select_by_index(ridxs, invert=True)

    # add random noise
    ridxs = np.random.choice(len(pcd.points), K, replace=False)
    noise_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(
        np.asarray(pcd.points)[ridxs] + voxel_size * np.random.randn(K, 3)))
    pcd += noise_pcd

    target_ids = np.concatenate([target_ids, np.zeros(K)])

    return pcd, target_ids 

def compute_max_r(obj):
    r_x = max(abs(obj.get_center()[0] - obj.get_max_bound()[0]),
              abs(obj.get_center()[0] - obj.get_min_bound()[0]))
    r_y = max(abs(obj.get_center()[1] - obj.get_max_bound()[1]),
              abs(obj.get_center()[1] - obj.get_min_bound()[1]))
    max_r = max(r_x, r_y)
    return max_r

# mushroom template
mushroom_filename = "./templates/mushrooms/mushroom_basic.obj"
global_cap_zthres = .25


#number_of_mushrooms = np.random.randint(5, 15)
position_bias = 2.00
scale_min = .5
scale_max = 1.5
#scale_min = 1.0
#scale_max = 1.5

#rotation_threshold = np.pi/8
rotation_threshold = np.pi/4

mushroom_mesh = o3d.io.read_triangle_mesh(mushroom_filename, True)
#mushroom_mesh = mushroom_mesh.simplify_quadric_decimation(20000)
mushroom_mesh = mushroom_mesh.remove_unreferenced_vertices()
mushroom_mesh = mushroom_mesh.remove_duplicated_vertices()
#mushroom_mesh = mushroom_mesh.simplify_vertex_clustering(30000)
mushroom_mesh.compute_vertex_normals()
mushroom_mesh.scale(0.005, center=mushroom_mesh.get_center())
mushroom_mesh.translate((0, 0, 0.3), relative=False)


# distractors !!!
cube_mesh = o3d.geometry.TriangleMesh().create_box()
cube_mesh.scale(0.1, center=cube_mesh.get_center())
cube_mesh.translate((0, 0, -.01), relative=False)
cone_mesh = o3d.geometry.TriangleMesh().create_cone()
cone_mesh.scale(0.1, center=cone_mesh.get_center())


from matplotlib.pyplot import cm
cmap = cm.get_cmap('tab10')



ground_path = "./templates/ground/"
ground_files = [ground_path+filename for filename in os.listdir(ground_path) if filename.endswith('.ply')]

def single_mushroom_feats(voxel_size=0.05):

    #voxel_size = np.random.uniform(.9, 1.1) * voxel_size

    template_mesh = copy.deepcopy(mushroom_mesh)
    template_mesh.translate((0, 0, 1 * 0.3), relative=False)

    mushroom_pcd = mushroom_mesh.sample_points_uniformly(10000)
    mushroom_pcd = mushroom_pcd.voxel_down_sample(voxel_size=voxel_size)


    pp = np.asarray(mushroom_pcd.points)
    # hardcoded threshold for selecting only the upper part
    ids = np.where(pp[:, -1] > global_cap_zthres)[0]

    # use updated points
    #pp = np.asarray(mushroom_pcd.points)
    normalized_height = (pp[:, -1] - pp[:, -1].min()) / (pp[:, -1].max() - pp[:, -1].min())

   #mushroom_pcd.scale(scale, center=mushroom_pcd.get_center())
    mushroom_pcd.translate((0, 0, 1 * 0.3), relative=False)
    #mushroom_pcd.rotate(R, center=mushroom_pcd.get_center())
    #mushroom_pcd.translate((x, y, 0))

    # tlabels = np.zeros(len(mushroom_pcd.points))

    labels = np.zeros(pp.shape[0])
    labels[ids] = 1

    conf = -np.ones(len(normalized_height))
    conf[ids] = normalized_height[ids]

    ctarget = np.asarray(mushroom_pcd.points)[ids].mean(axis=0)

    features = np.concatenate([pp[ids] - ctarget, conf[ids].reshape(-1, 1), labels[ids].reshape(-1, 1)], axis=-1)

    return pp[ids], features, template_mesh


def random_hidden_points_removal(tmp_pcd):

    camera_location = [0.0, 0.0, .5] + np.random.uniform(.0, .5, size=[3])
    camera_location = 10 * camera_location / np.linalg.norm(camera_location)

    _, idxs =tmp_pcd.hidden_point_removal(camera_location, 10 * np.random.uniform(50, 500))

    #print(len(idxs), len(scene_pcd.points))

    idxs = np.sort(idxs)
    tmp_pcd = tmp_pcd.select_by_index(idxs)

    return tmp_pcd, idxs


def scene_generation(number_of_mushrooms, voxel_size=0.05, loosen_value = .85, add_distractors=True):
    bvoxel_size = np.random.uniform(0.7, 1.25) * voxel_size

    cultivation = []

    scene_pcd = o3d.geometry.PointCloud()
    labels = []
    instances = []
    conf = []
    ctargets = []
    rot_vecs = []

    bboxes = []

    tground_pcd = o3d.io.read_point_cloud(ground_files[np.random.randint(len(ground_files))])
    if add_distractors:
        for i in range(np.random.randint(1, 5)):
            if np.random.uniform() > .3:
                tmesh = copy.deepcopy(cube_mesh)
            else:
                tmesh = copy.deepcopy(cone_mesh)


            rotv = np.pi * np.random.randn(3)

            R = tmesh.get_rotation_matrix_from_xyz(rotv)
            tmesh.rotate(R, center=tmesh.get_center())

            scale=np.random.uniform(.5, 20.0)
            tmesh.scale(scale, center=tmesh.get_center())

            xy = position_bias * np.random.uniform(-1, 1, size=2)
            tmesh.translate(list(xy) + [-scale * 0.01])


            tground_pcd += tmesh.sample_points_uniformly(int(scale * np.random.randint(100, 500)))

    scales = []
    rotations = []
    translations = []
    cnt = 0
    #for i, mushroom in enumerate(cultivation):
    while cnt < number_of_mushrooms:

        #tt = time.time()

        mushroom = copy.deepcopy(mushroom_mesh)


        scale = round(random.uniform(scale_min, scale_max), 2)
        mushroom.scale(scale, center=mushroom.get_center())
        scales.append(scale)
        mushroom.translate((0, 0, scale * 0.3), relative=False)

        rotv = np.asarray([rotation_threshold, rotation_threshold, np.pi]) * np.random.uniform(-1, 1, size=3)
        R = mushroom.get_rotation_matrix_from_xyz(rotv)
        mushroom.rotate(R, center=mushroom.get_center())
        rotations.append(rotv)

        # Choose random position : translation
        x = round(random.uniform(-position_bias, position_bias), 2)
        y = round(random.uniform(-position_bias, position_bias), 2)
        mushroom.translate((x, y, 0))
        translations.append([x, y])
        r_max = compute_max_r(mushroom)

        no_collision = True
        for j, previous_mushroom in enumerate(cultivation):
            if np.linalg.norm(mushroom.get_center() - previous_mushroom.get_center()) < \
               loosen_value * (compute_max_r(previous_mushroom) + r_max):
                no_collision = False
                break

        if no_collision:
            cultivation.append(mushroom)

            N = np.random.randint(2000, 10000)
            tvoxel_size = np.random.uniform(.75, 1.25) * bvoxel_size / scale
            # sample N points
            mushroom_pcd = mushroom_mesh.sample_points_uniformly(N)
            mushroom_pcd = mushroom_pcd.voxel_down_sample(voxel_size=tvoxel_size)

            pp = np.asarray(mushroom_pcd.points)
            # hardcoded threshold for selecting only the upper part
            ids = np.where(pp[:, -1] > global_cap_zthres)[0]

            mushroom_pcd, tids = augment_template(mushroom_pcd, ids, voxel_size=tvoxel_size)

            # use updated points
            pp = np.asarray(mushroom_pcd.points)
            normalized_height = (pp[:, -1] - pp[tids==1, -1].min()) / (pp[tids==1, -1].max() - pp[tids==1, -1].min())


            mushroom_pcd.scale(scale, center=mushroom_pcd.get_center())

            mushroom_pcd.points = o3d.utility.Vector3dVector(np.asarray(mushroom_pcd.points) * np.random.uniform(.75, 1.25, size=(1,3)))

            mushroom_pcd.translate((0, 0, scale * 0.3), relative=False)
            mushroom_pcd.rotate(R, center=mushroom_pcd.get_center())
            mushroom_pcd.translate((x, y, 0))

            rot_vecs += [R[:, -1].reshape(1, -1)]
            ctarget = np.asarray([x, y, scale * 0.3])
            #ctarget =

            pp = np.asarray(mushroom_pcd.points)
            mtmp = o3d.geometry.PointCloud()
            mtmp.points = o3d.utility.Vector3dVector(pp[tids==1])
            mbbox = mtmp.get_oriented_bounding_box()
            mbbox.color = [0, 1, 0]
            bboxes += [mbbox]

            if np.random.uniform() > 0.5:
                #mushroom_pcd, nids = random_hidden_points_removal(mushroom_pcd, tids)
                mushroom_pcd, nids = random_hidden_points_removal(mushroom_pcd)
            else:
                nids = np.arange(0, len(tids))



            #tlabels = np.zeros(len(mushroom_pcd.points))

            labels += [tids[nids]]
            #instances += [(cnt+1) * np.ones(len(tids))]
            instances += [(cnt + 1) * np.ones(len(nids))]

            tconf = -np.ones(len(normalized_height))
            tconf[tids==1] = normalized_height[tids==1]
            conf += [tconf[nids]]

            #ctargets += [np.asarray(mushroom_pcd.points)[tids==1].mean(axis=0)]

            ctargets += [ctarget]

            #mushroom_pcd.paint_uniform_color(cmap(cnt)[:3])
            scene_pcd += mushroom_pcd

            cnt += 1

    ground_pcd = tground_pcd.voxel_down_sample(voxel_size=np.random.uniform(.8, 1.2) * bvoxel_size)

    scene_pcd += ground_pcd


    scene_points = np.asarray(scene_pcd.points)
    scene_points += .1 * voxel_size * np.random.randn(scene_points.shape[0], 3)
    scene_pcd.points = o3d.utility.Vector3dVector(np.asarray(scene_points))

    labels += [np.zeros(len(ground_pcd.points))]
    labels = np.concatenate(labels)

    instances += [np.zeros(len(ground_pcd.points))]
    instances = np.concatenate(instances)

    conf += [-np.ones(len(ground_pcd.points))]
    conf = np.concatenate(conf)

    # super fine-tweaked camera locations
    if np.random.rand() > .5:
        camera_location = [0.0, 0.0, .5] + np.random.uniform(.0, .5, size=[3])
        camera_location = 10 * camera_location / np.linalg.norm(camera_location)

        _, idxs = scene_pcd.hidden_point_removal(camera_location, 1e4)

        #print(len(idxs), len(scene_pcd.points))

        idxs = np.sort(idxs)
        scene_pcd = scene_pcd.select_by_index(idxs)

        # final random scale!
        #scene_pcd.scale(np.random.uniform(.95, 1.05), center=scene_pcd.get_center())

        labels = labels[idxs]
        instances = instances[idxs]
        conf = conf[idxs]


    #print(time.time() - tt)

    #label_color = np.asarray([[.1, .1, .1] if label==0 else [1, .0, .0] for label in labels])
    #scene_pcd.colors = o3d.utility.Vector3dVector(label_color)


    ttscale = .1 * np.random.uniform(.75, 1.25)
    scene_pcd.points = o3d.utility.Vector3dVector(ttscale * np.asarray(scene_pcd.points))
    ctargets = ttscale * np.asarray(ctargets)

    # random remove 5 - 20% of points uniformly
    Np = len(scene_pcd.points)
    Nc = np.random.randint(int(.05 * Np), int(.2 * Np))
    rinds = np.random.choice(Np, Nc, replace=False)
    # find complement indices
    cinds = np.setdiff1d(np.arange(Np), rinds)

    scene_pcd = scene_pcd.select_by_index(rinds, invert=True)
    labels = labels[cinds]
    instances = instances[cinds]
    conf = conf[cinds]

    bboxes = [bbox.scale(ttscale, center=[0,0,0]) for bbox in bboxes]
    #scene_pcd = scene_pcd.voxel_down_sample(voxel_size=voxel_size)

    #o3d.visualization.draw_geometries([scene_pcd])

    return scene_pcd, rot_vecs, bboxes, labels, instances, ctargets, conf

'''
scene_pcd, _, bboxes, labels, _, _, conf = scene_generation(40, voxel_size=0.025, loosen_value=.6)
#label_color = np.asarray([[.1, .1, .1] if label==0 else [1, .0, .0] for label in labels])
#scene_pcd.colors = o3d.utility.Vector3dVector(label_color)
#o3d.visualization.draw_geometries([scene_pcd] + bboxes)

label_color = np.asarray([[.2, .2, .8] if c==-1 else [c, .0, .0] for c in conf])
scene_pcd.colors = o3d.utility.Vector3dVector(label_color)
o3d.visualization.draw_geometries([scene_pcd] + bboxes)

print(len(scene_pcd.points))

voxel_size = 0.005
npcd, point_inds, _ = scene_pcd.voxel_down_sample_and_trace(voxel_size=voxel_size,
                                                                 min_bound=scene_pcd.get_min_bound(),
                                                                 max_bound=scene_pcd.get_max_bound())

print(len(npcd.points))
point_inds = point_inds.max(axis=1)

label_color = np.asarray([[.2, .2, .8] if c==-1 else [c, .0, .0] for c in conf[point_inds]])
npcd.colors = o3d.utility.Vector3dVector(label_color)
o3d.visualization.draw_geometries([npcd] + bboxes)

'''