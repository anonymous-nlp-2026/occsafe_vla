import numpy as np

def voxel_to_bev(semantic_3d, binary_3d):
    bev_binary = binary_3d.any(axis=2)
    bev_semantic = semantic_3d.max(axis=2)
    return bev_semantic, bev_binary
