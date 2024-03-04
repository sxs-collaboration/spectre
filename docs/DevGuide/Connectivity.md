\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond

# Visualisation Connectivity {#visualisation_connectivity}

This guide is split into two parts. The first part details what is meant by
'connectivity' and how it is used to visualise SpECTRE simulations in standard
visualisation software such as Paraview. The second part details the
methodology associated with ExtendConnectivity - a post-processing executable
that removes gaps in the default visualisations produced by SpECTRE.

## What is Connectivity?

SpECTRE simulations are built on a computational grid composed of individual
gridpoints. The 'connectivity' is a list added in the output simulation files
that instructs visualisation software, namely Paraview, on how to connect these
gridpoints to create lines/surfaces/volumes (depending on the dimension of the
simulation) as needed. Given these instructions, Paraview interpolates the data
over the regions between gridpoints to create line/surface/volume meshes.

In general, the connectivity list contains gridpoints in a particular sequence
that corresponds to a line/surface/volume mesh in Paraview. These gridpoints are
labelled with integers that are based on the order in which they appear in the
h5 files output by the simulation. As one might expect, this list is structured
differently depending on the dimension of the simulation. The key underlying
difference is the size of the smallest connected region. In 1D, our domain is
composed of lines so the smallest connected region is a line made up by two
points. In 2D, the smallest connected region used by SpECTRE is typically a
quadrilateral made up by four points. In 3D, the smallest connected region used
by SpECTRE is typically a hexahedron made up by eight points. A visual
representation is illusutrated below. (Note: The numbering of points has been
chosen arbitrarily. In reality, this numbering is determined based on the order
of gridpoints in the h5 file(s)).

\image html DefaultConnectivity.png "Basic connectivity in 1D, 2D, and 3D"

The figure above shows how connectivity is defined for a region in each
dimension. Fig (a) shows the gridpoint structure for these regions and Fig (c)
shows how the resulting mesh appears. Fig (b) shows how a connectivity list
should be sequenced to create the desired mesh. Then, in 1D, our connectivity
list to create the line in (c) would be [1,2]. Similarly, in 2D, our
connectivity list would be [1,2,4,3] and in 3D, our list would be
[1,2,4,3,5,6,8,7]. (Note that this sequencing is dictated by Paraview's
conventions, not SpECTRE's.)

The above example illustrates how to connect the smallest connected region in
each dimension. The complete connectivity list, detailing how the entire domain
mesh is to be created, is made by concatenating individual connectivity lists.
Thus, in 1D, Paraview interprets the list [1,2,2,3] in blocks of 2, with the
first block [1,2] representing a line between gridpoints 1 and 2, and the second
block [2,3] representing a line between gridpoints 2 and 3. This works similarly
in the other dimensions.

