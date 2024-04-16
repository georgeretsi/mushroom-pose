import numpy as np
import open3d as o3d
import torch
import torch.nn as nn


from create_scene import scene_generation
from models.utils import process_input


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


name_prefix = 'fnl'

from models.resunet import ResUNet3D

# model_path = 'ResUNetBN2C-32feat-3DMatch.pth'
# checkpoint = torch.load(model_path)
# backbone_model.load_state_dict(checkpoint['state_dict'])
model = ResUNet3D(1, 5)
#odel_path = './models/ResUNetBN2C-32feat-3DMatch.pth'
#checkpoint = torch.load(model_path)
#model.backbone.load_state_dict(checkpoint['state_dict'])
model.load_state_dict(torch.load('fnl_model.pt'))
model = model.cuda()



Niter = 60000
update_every_n = 1
Kdisplay = 100
voxel_size = 0.005

optimizer = torch.optim.Adam(list(model.parameters()), 1e-3)
#optimizer = torch.optim.Adam(list(models.parameters()), 1e-3)
optimizer.zero_grad()
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [Niter//2])

tloss = .0

search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=30)
for i in range(Niter):

    #lower_bound = 5
    upper_bound = 5 +  1 + int(50 * i / Niter) # up to 45 mushrooms in a scene in linear progression
    lower_bound = max(5, upper_bound - 10)
    #upper_bound = 50


    scene_pcd, rot_vecs, _, labels, instances, ctargets, confs = scene_generation(np.random.randint(lower_bound, upper_bound),
                                                                        loosen_value=np.random.uniform(.6, .8),
                                                                        voxel_size=10 * np.random.uniform(.8, 1.2) * voxel_size)

    points = torch.Tensor(np.asarray(scene_pcd.points)).cuda()
    #with torch.no_grad():
    tinput, point_inds, _ = process_input(xyz=np.asarray(scene_pcd.points),
                                          voxel_size=np.random.uniform(.9, 1.1) * voxel_size,
                                          device=device)

    #reduced_points = tinput.C[:, :3]


    encoded_features = model(tinput)

    pred_center = encoded_features[..., :3]
    pred_orient = encoded_features[..., -1].squeeze()
    pred_class = encoded_features[..., -2].squeeze()

    points = points[point_inds]
    labels = labels[point_inds]
    instances = instances[point_inds]
    confs = confs[point_inds]

    cd = torch.cdist(points[labels > 0].unsqueeze(0), torch.Tensor(ctargets).cuda().unsqueeze(0)).squeeze().topk(2, largest=False, dim=1)[0]


    targets = np.asarray([ctargets[int(c) - 1] if l == 1 else [0, 0, 0] for c, l in zip(instances, labels)])
    targets = torch.Tensor(targets).cuda()

    labels = torch.Tensor(labels).cuda()
    labels[labels > 0] *= (torch.abs(1 - cd[:, 0]/cd[:, 1]) > .25)


    ftargets = (points - targets).cuda()
    ftargets[labels==0] *= 0 # zero difference when no mushroom!!

    ctargets = torch.Tensor(confs).cuda()

    lscale = 10 # * i / Niter
    loss = 5.0 * torch.nn.BCEWithLogitsLoss()(pred_class, labels) + \
           lscale * torch.nn.MSELoss()(10 * pred_center[labels==1], 10 * ftargets[labels==1]) + \
           .1 * torch.nn.MSELoss()(pred_center[labels==0], ftargets[labels==0]) + \
           10.0 * torch.nn.MSELoss()(pred_orient[labels == 1], ctargets[labels == 1])

    tloss += loss.item()

    loss.backward()

    torch.nn.utils.clip_grad_norm(list(model.parameters()), .1)

    if i % update_every_n == 0:
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    if i%Kdisplay == 0:
        print(tloss/ Kdisplay)
        tloss = 0.0

        torch.save(model.cpu().state_dict(), name_prefix + '_model.pt')
        model.cuda()

torch.save(model.cpu().state_dict(), name_prefix + '_model.pt')
