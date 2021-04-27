# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def reconstruct(u, extents, dim, ghost_zones, func):
    # Need to push back extra vars to get test to be generic.
    number_of_dims = len(extents)
    if len(extents) < 3:
        for i in range(len(extents), 3):
            extents.append(1)

    u = np.reshape(np.copy(np.asarray(u)), extents, order='F')
    recons_x_lower_face = []
    recons_x_upper_face = []
    # Reconstruction in x
    for k in range(ghost_zones[2] + 1, extents[2] - ghost_zones[2] -
                   1) if number_of_dims > 2 else range(1):
        for j in range(ghost_zones[1] + 1, extents[1] - ghost_zones[1] -
                       1) if number_of_dims > 1 else range(1):
            recons_upper_of_cell = []
            recons_lower_of_cell = []
            for i in range(ghost_zones[0], extents[0] - ghost_zones[0]):
                func(recons_upper_of_cell, recons_lower_of_cell, u, i, j, k, 0)
            # By deleting the first entry in lower_of_cell and the last
            # entry of upper_of_cell we convert to the face-based indices
            recons_lower_of_cell.pop(0)
            recons_upper_of_cell.pop(-1)
            recons_x_lower_face = recons_x_lower_face + recons_upper_of_cell
            recons_x_upper_face = recons_x_upper_face + recons_lower_of_cell

    if dim == 1:
        return [[recons_x_lower_face], [recons_x_upper_face]]

    # Reconstruction in y
    recons_y_lower_face = []
    recons_y_upper_face = []
    for k in range(ghost_zones[2] + 1, extents[2] - ghost_zones[2] -
                   1) if number_of_dims > 2 else range(1):
        recons_upper_of_cell = []
        recons_lower_of_cell = []
        for j in range(ghost_zones[1], extents[1] - ghost_zones[1]):
            for i in range(ghost_zones[0] + 1,
                           extents[0] - ghost_zones[0] - 1):
                func(recons_upper_of_cell, recons_lower_of_cell, u, i, j, k, 1)
        # By deleting the first entry in lower_of_cell and the last
        # entry of upper_of_cell we convert to the face-based indices
        for i in range(ghost_zones[0] + 1, extents[0] - ghost_zones[0] - 1):
            recons_lower_of_cell.pop(0)
            recons_upper_of_cell.pop(-1)
        recons_y_lower_face = recons_y_lower_face + recons_upper_of_cell
        recons_y_upper_face = recons_y_upper_face + recons_lower_of_cell
    if dim == 2:
        return [[recons_x_lower_face, recons_y_lower_face],
                [recons_x_upper_face, recons_y_upper_face]]

    # Reconstruction in z
    recons_z_lower_face = []
    recons_z_upper_face = []
    for k in range(ghost_zones[2], extents[2] - ghost_zones[2]):
        for j in range(ghost_zones[1] + 1, extents[1] - ghost_zones[1] - 1):
            for i in range(ghost_zones[0] + 1,
                           extents[0] - ghost_zones[0] - 1):
                func(recons_z_lower_face, recons_z_upper_face, u, i, j, k, 2)
    # By deleting the first entry in lower_of_cell and the last
    # entry of upper_of_cell we convert to the face-based indices
    for j in range(ghost_zones[1] + 1, extents[1] - ghost_zones[1] - 1):
        for i in range(ghost_zones[0] + 1, extents[0] - ghost_zones[0] - 1):
            recons_z_upper_face.pop(0)
            recons_z_lower_face.pop(-1)

    return [[recons_x_lower_face, recons_y_lower_face, recons_z_lower_face],
            [recons_x_upper_face, recons_y_upper_face, recons_z_upper_face]]
