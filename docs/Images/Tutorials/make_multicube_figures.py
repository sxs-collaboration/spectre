#!/usr/bin/env python

# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


plt.rcParams['figure.figsize'] = (4, 4)
plt.rcParams['font.size'] = 10
# higher-quality fonts, but much slower to render:
# plt.rcParams['font.serif'] = 'Computer Modern'
# plt.rcParams['text.usetex'] = True


def draw_basis_vectors(ax, x_origin, label_permutation=[0, 1, 2],
                       flip_vector=[False, False, False],
                       use_greek_labels=True):
    from matplotlib.patches import FancyArrowPatch
    from mpl_toolkits.mplot3d import proj3d

    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)

    length = 0.3
    labels = [r'$\xi$', r'$\eta$',
              r'$\zeta$'] if use_greek_labels else ['$x$', '$y$', '$z$']
    arrow_props = dict(mutation_scale=10, arrowstyle='-|>', color='k',
                       shrinkA=0, shrinkB=0)
    x0, y0, z0 = [x_origin[i] + (length if flip_vector[i] else 0.0)
                  for i in range(3)]
    dx, dy, dz = [length * (-1.0 if flip_vector[i] else 1.0) for i in range(3)]
    ax.add_artist(Arrow3D([x0, x0 + dx], [y0, y0], [z0, z0], **arrow_props))
    ax.add_artist(Arrow3D([x0, x0], [y0, y0 + dy], [z0, z0], **arrow_props))
    ax.add_artist(Arrow3D([x0, x0], [y0, y0], [z0, z0 + dz], **arrow_props))
    dxt, dyt, dzt = [-length - 0.15 if flip_vector[i] else length + 0.05
                     for i in range(3)]
    dyt = dyt - (0.15 if flip_vector[1] else 0.0)
    ax.text(x0 + dxt, y0, z0, labels[label_permutation[0]])
    ax.text(x0, y0 + dyt, z0, labels[label_permutation[1]])
    ax.text(x0, y0, z0 + dzt, labels[label_permutation[2]])


def draw_polygon_collection(ax, polygons, face_colors, edge_colors):
    collection = Poly3DCollection(polygons)
    collection.set_color(face_colors)
    collection.set_edgecolor(edge_colors)
    collection.set_linewidth(0.6)
    ax.add_collection3d(collection)


def quad(p, i, j, k, l):
    return [p[i], p[j], p[k], p[l]]


def cube(p, index_start, strides):
    sx, sy, sz = strides

    def quad_x_normal(i):
        return quad(p, i, i + sy, i + sy + sz, i + sz)

    def quad_y_normal(i):
        return quad(p, i, i + sx, i + sx + sz, i + sz)

    def quad_z_normal(i):
        return quad(p, i, i + sx, i + sx + sy, i + sy)

    quad_lower_x = quad_x_normal(index_start)
    quad_upper_x = quad_x_normal(index_start + sx)
    quad_lower_y = quad_y_normal(index_start)
    quad_upper_y = quad_y_normal(index_start + sy)
    quad_lower_z = quad_z_normal(index_start)
    quad_upper_z = quad_z_normal(index_start + sz)

    return [quad_lower_x, quad_upper_x, quad_lower_y, quad_upper_y,
            quad_lower_z, quad_upper_z]


def black_color(mean_xyz):
    return 'k'


class CubeManager:
    def __init__(self, point_list, strides, edge_color_function=black_color):
        self._points = point_list
        self._strides = strides
        self._ecf = edge_color_function
        self.quads = []
        self.face_colors = []
        self.edge_colors = []

    def add_cube(self, start_index, these_faces_transparent=[]):
        c = cube(self._points, start_index, self._strides)
        self.quads.extend(c)
        clear = (1, 1, 1, 0)
        white_semi_transparent = (1, 1, 1, 0.75)
        self.face_colors.extend([clear if face in these_faces_transparent
                                 else white_semi_transparent
                                 for face in range(6)])
        self.edge_colors.extend([self._ecf(np.average(face, axis=0))
                                 for face in c])


def one_cube_figure():
    p = np.array([[x, y, z] for z in [0, 1] for y in [0, 1] for x in [0, 1]])
    cubes = CubeManager(p, strides=[1, 2, 4])
    cubes.add_cube(0)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)

    draw_polygon_collection(ax, cubes.quads, cubes.face_colors,
                            cubes.edge_colors)

    draw_basis_vectors(ax, [0.1, 0.1, 0.1])
    for i in range(p.shape[0]):
        # shift labels of upper-y face to avoid overlaps
        shift_y = (-0.25 if p[i, 1] < 0.5 else 0.1)
        ax.text(p[i, 0], p[i, 1] + shift_y, p[i, 2], '$'+str(i)+'$')

    fig.savefig('onecube_numbered.png')


def two_cube_figures():
    p = np.array([[x, y, z] for z in [0, 1] for y in [0, 1]
                 for x in [0, 1, 2]])
    cubes = CubeManager(p, strides=[1, 3, 6])
    cubes.add_cube(0, [1])
    cubes.add_cube(1, [0])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(-0.5, 1.5)
    ax.set_zlim(-0.5, 1.5)

    draw_polygon_collection(ax, cubes.quads, cubes.face_colors,
                            cubes.edge_colors)

    draw_basis_vectors(ax, [0.1, 0.1, 0.1])
    draw_basis_vectors(ax, [1.1, 0.1, 0.1], label_permutation=[2, 0, 1])

    fig.savefig('twocubes.png')

    for i in range(p.shape[0]):
        # shift labels of upper-y face to avoid overlaps
        shift_y = (-0.25 if p[i, 1] < 0.5 else 0.1)
        ax.text(p[i, 0], p[i, 1] + shift_y, p[i, 2], '$'+str(i)+'$')

    fig.savefig('twocubes_numbered.png')


def eight_cube_figures():
    p = np.array([[x, y, z] for z in [0, 1, 2] for y in [0, 1, 2]
                 for x in [0, 1, 2]])

    def red_black_blue(mean_xyz):
        r = max(1.0 - mean_xyz[1], 0.0)
        b = max(mean_xyz[1] - 1.0, 0.0)
        return [r, 0.0, b]

    cubes = CubeManager(p, strides=[1, 3, 9],
                        edge_color_function=red_black_blue)
    cubes.add_cube(0, [1, 3, 5])
    cubes.add_cube(1, [0, 3, 5])
    cubes.add_cube(3, [1, 2, 5])
    cubes.add_cube(4, [0, 2, 5])
    cubes.add_cube(9, [1, 3, 4])
    cubes.add_cube(10, [0, 3, 4])
    cubes.add_cube(12, [1, 2, 4])
    cubes.add_cube(13, [0, 2, 4])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 2.0)
    ax.set_zlim(0.0, 2.0)

    draw_polygon_collection(ax, cubes.quads, cubes.face_colors,
                            cubes.edge_colors)

    fig.savefig('eightcubes.png')

    draw_basis_vectors(ax, [0.55, -0.75, 0.0], use_greek_labels=False)
    for i in range(p.shape[0]):
        # shift labels on each y=constant surface, to reduce overlaps
        shift = [0.0, 0.0, 0.0]
        if p[i, 1] == 0.0:
            shift[0] = -0.05
            shift[1] = -0.3
        elif p[i, 1] == 1.0:
            shift[0] = 0.02
            shift[2] = -0.2
        else:
            shift[1] = 0.1
        # this label needs additional manual tweaking
        if i == 15:
            shift[0] = -0.2
            shift[1] = -0.1
        # reduce R,B components so colors aren't too bright
        textcolor = red_black_blue(p[i])
        textcolor[0] /= 1.5
        textcolor[2] /= 1.5
        ax.text(p[i, 0] + shift[0], p[i, 1] + shift[1], p[i, 2] + shift[2],
                '$'+str(i)+'$', color=textcolor)

    fig.savefig('eightcubes_numbered.png')


def eight_cube_rotated_exploded_figure():
    p = np.array([[x, y, z] for z in [0, 1, 2, 3] for y in [0, 1, 2, 3]
                 for x in [0, 1, 2, 3]])

    def red_black_blue(mean_xyz):
        y_normalization = 1.5
        r = max(1.0 - mean_xyz[1] / y_normalization, 0.0)
        b = max(mean_xyz[1] / y_normalization - 1.0, 0.0)
        return [r, 0.0, b]

    cubes = CubeManager(p, strides=[1, 4, 16],
                        edge_color_function=red_black_blue)
    cubes.add_cube(0)
    cubes.add_cube(2)
    cubes.add_cube(8)
    cubes.add_cube(10)
    cubes.add_cube(32)
    cubes.add_cube(34)
    cubes.add_cube(40)
    cubes.add_cube(42)

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 3.0)
    ax.set_zlim(0.0, 3.0)

    draw_polygon_collection(ax, cubes.quads, cubes.face_colors,
                            cubes.edge_colors)

    draw_basis_vectors(ax, [1.0, -0.8, 0.0], use_greek_labels=False)
    draw_basis_vectors(ax, [0.17, 0.17, 0.25])
    draw_basis_vectors(ax, [2.17, 0.17, 0.25], label_permutation=[2, 1, 0],
                       flip_vector=[False, False, True])
    draw_basis_vectors(ax, [0.17, 2.17, 0.45], label_permutation=[0, 2, 1],
                       flip_vector=[False, False, True])
    draw_basis_vectors(ax, [2.17, 2.17, 0.3], label_permutation=[2, 0, 1],
                       flip_vector=[False, True, True])
    draw_basis_vectors(ax, [0.17, 0.17, 2.25], label_permutation=[1, 0, 2],
                       flip_vector=[False, True, False])
    draw_basis_vectors(ax, [2.17, 0.17, 2.25], label_permutation=[1, 2, 0],
                       flip_vector=[False, True, True])
    draw_basis_vectors(ax, [0.17, 2.17, 2.3], label_permutation=[2, 0, 1],
                       flip_vector=[False, True, True])
    draw_basis_vectors(ax, [2.17, 2.17, 2.45])
    fig.savefig('eightcubes_rotated_exploded.png')


def tesseract_figure():
    p = np.array([[x, y, z] for z in [0, 1, 2, 3, 4] for y in [0, 1, 2, 3]
                 for x in [0, 1, 2, 3]])

    def red_black_blue(mean_xyz):
        # note: the tesseract is a bigger domain, so mean_xyz values can be
        # larger. the factor of 1.5 normalizes R,B values back into [0,1].
        y_normalization = 1.5
        r = max(1.0 - mean_xyz[1]/y_normalization, 0.0)
        b = max(mean_xyz[1]/y_normalization - 1.0, 0.0)
        return [r, 0.0, b]

    cubes = CubeManager(p, strides=[1, 4, 16],
                        edge_color_function=red_black_blue)
    cubes.add_cube(5, [5])
    cubes.add_cube(21, [4, 5])
    cubes.add_cube(33, [3])
    cubes.add_cube(36, [1])
    cubes.add_cube(38, [0])
    cubes.add_cube(41, [2])
    cubes.add_cube(53, [4])

    # also draw the back 3 sides of the bounding box
    box_quads = np.array([quad(p, 0, 12, 76, 64),
                          quad(p, 12, 15, 79, 76),
                          quad(p, 0, 3, 15, 12)])

    fig = plt.figure(figsize=(6, 6))
    ax = Axes3D(fig)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_zlim(0.0, 4.0)

    # draw bounding box first, else matplotlib gets the layer ordering wrong
    draw_polygon_collection(ax, box_quads, 'white', 'lightgrey')
    draw_polygon_collection(ax, cubes.quads, cubes.face_colors,
                            cubes.edge_colors)

    # these are the visible points on the tesseract
    points_to_label = [5, 6, 10, 21, 22, 26, 36, 33, 34, 38, 39, 43, 52, 49,
                       50, 53, 54, 55, 56, 58, 59, 62, 69, 70, 73, 74]
    # these are the points we want to show from the bounding box
    points_to_label.extend([0, 3, 15, 64, 76, 79])
    for i in points_to_label:
        # shift labels on each y=constant surface, to reduce overlaps
        shift = [0.0, 0.0, 0.0]
        if p[i, 1] == 0.0 or p[i, 1] == 1.0:
            shift[0] = -0.07
            shift[1] = -0.37
        else:
            shift[1] = 0.1
        # specific individual corrections at complex tesseract corners
        if i == 38:
            shift[2] = 0.27
        elif i == 39:
            shift[0] = 0.15
            shift[1] = -0.2
        elif i == 53:
            shift[0] = 0.24
        elif i == 54:
            shift[0] = 0.24
        elif i == 58:
            shift[2] = -0.3
        elif i == 69:
            shift[2] = 0.15
        # reduce R,B components so colors aren't too bright
        textcolor = red_black_blue(p[i])
        textcolor[0] /= 1.5
        textcolor[2] /= 1.5
        ax.text(p[i, 0] + shift[0], p[i, 1] + shift[1], p[i, 2] + shift[2],
                '$'+str(i)+'$', color=textcolor)

    fig.savefig('tesseract_numbered.png')


one_cube_figure()
two_cube_figures()
eight_cube_figures()
eight_cube_rotated_exploded_figure()
tesseract_figure()
