import torch
import numpy as np

import MinkowskiEngine as ME

def process_input(xyz, rgb=None, normal=None, voxel_size=0.05, device=None):

  if device is None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

  feats = []
  if rgb is not None:
    # [0, 1]
    feats.append(rgb - 0.5)

  if normal is not None:
    # [-1, 1]
    feats.append(normal / 2)

  if rgb is None and normal is None:
    feats.append(np.ones((len(xyz), 1)))

  feats = np.hstack(feats)

  coords = np.floor(xyz / voxel_size)
  coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
  point_inds = inds.detach().cpu().numpy()

  # Convert to batched coords compatible with ME
  coords = ME.utils.batched_coordinates([coords])

  feats = feats[point_inds]
  feats = torch.tensor(feats, dtype=torch.float32)
  coords = torch.tensor(coords, dtype=torch.int32)

  stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

  return stensor, point_inds, feats
