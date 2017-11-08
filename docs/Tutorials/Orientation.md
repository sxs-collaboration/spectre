\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Orientation {#tutorial_orientations}

### %Orientations between %Blocks
The Orientations between Blocks are used to properly communicate fluxes
across boundaries between adjacent Blocks that do not necessarily have their
logical axes aligned. The `Orientation` class is used to keep track of which
pair of axes in a pair of adjacent Blocks lie along the same physical
direction at a boundary.

### Algorithms for determining the %Orientation of %Blocks given corners
DomainHelpers is a collection of algorithms that are used to determine the
Orientations between Blocks in a Domain using a set of corner numbering
schemes. The corner numberings must be determined and provided by the user.
This tutorial will explain the corner numbering schemes used, and how to
determine the correct corner numbering for a Domain. We assume the user has
a Domain that has that has already been partitioned
into Blocks in the form of a schematic diagram, and that the orientation of
the logical axes within each Block has been determined before proceeding with
this tutorial.

### Global Corner Numbering Scheme (Global CNS)
The partitioning of the Domain defines a global unordered set
of corners. For example, a cubical Domain which is partitioned equally into
two Blocks has 12 corners; although each Block has 8 corners, 4 are shared
among them. To assign a Global CNS to the Domain, one may arbitrarily assign
corner ids to each of the twelve corners in the Domain, so long as each corner
has a single, unique id.

\note This Global CNS assumes that there is no additional identifying of faces
with one another for periodic boundary conditions. That is, that no block has
itself as a neighbor. If you wish to additionally identify faces of the same
block with each other, that must be done in an additional step. This step is
explained in the "Setting Periodic Boundary Conditions" section.

### Local Corner Numbering Scheme (Local CNS)
The orientation of the logical axes in each Block define an ordered set of
corners. Once the logical axes in each Block are determined, it is possible
to write down a Local CNS:

Each Block has \f$2^{dim}\f$ local corners, numbered from \f$0\f$ to
\f$2^{dim} - 1\f$. We will take the three-dimensional cube as an example:
This Block has 8 local corners (regardless how many of these are
shared), numbered from 0 to 7. The corners are labelled according to the
CoordinateMap corresponding to the Block, as follows:

```
Corner 0 is the location of the lower xi, lower eta, lower zeta corner.
Corner 1 is the location of the upper xi, lower eta, lower zeta corner.
Corner 2 is the location of the lower xi, upper eta, lower zeta corner.
Corner 3 is the location of the upper xi, upper eta, lower zeta corner.
Corner 4 is the location of the lower xi, lower eta, upper zeta corner.
Corner 5 is the location of the upper xi, lower eta, upper zeta corner.
Corner 6 is the location of the lower xi, upper eta, upper zeta corner.
Corner 7 is the location of the upper xi, upper eta, upper zeta corner.
```

   We can summarize this information:

```
Corner 0: 000
Corner 1: 001
Corner 2: 010
Corner 3: 011
Corner 4: 100
Corner 5: 101
Corner 6: 110
Corner 7: 111
```

Where 0 and 1 indicate lower and upper in the corresponding axis (zeta,
eta, xi), respectively, and the ordering has been reversed so that the
rightmost column corresponds to the xi position and the leftmost column
to the zeta position.

\note This Local CNS is independent of the Global CNS above.

### The Ordered Subset of the Global CNS (Subset CNS):
With the Global CNS in hand, each Block inherits a ordered subset of Global
CNS, the order determined by the ordering of the Local CNS, and the elements
of the set determined by how one assigned the Global CNS to the Domain.
For example, if in a Global CNS one had assigned the id "7" to the lower xi,
lower eta, lower zeta corner of a Block,
then the Subset CNS corresponding to this Block will begin as {7, ..., ...}.
The Subset CNS encodes the relative Orientations between
each Block.

The algorithms in DomainHelpers take these Subset CNSs as input, and from them
determines the proper relative orientations between blocks.

### Explanation of the Algorithms in DomainHelpers:

For illustrative purposes, we will use the following Domain composed of two
Blocks as described above as an example.
Because there are 12 corners in this Domain, we will arbitrarily assign a
unique id to each corner.
Knowing the orientation of the logical axes within a block, we construct a
Subset CNS for each Block.<br>
Here is one possible result, given some relative orientation between the
blocks:

```
Block1: {3,0,4,1,9,6,10,7}
Block2: {1,2,4,5,7,8,10,11}
```

The values of the ids only serve to identify which corners are unique and which
are shared. This is determined by the Global CNS. The order of the ids in the
list is determined by the Local CNS. We take advantage of the fact that the
array index of the global corner id is the number of the corner in the local
CNS.

The algorithm begins by determining the shared corners between the
Blocks:

```
result: {4,1,10,7}
```

The next step is to determine the local ids of these shared global ids:

```
Block1 result: {2,3,6,7}
Block2 result: {2,0,6,4}
```

 The next step is to convert these ids into their binary representation:

```
Block1 result: {010,011,110,111}
Block2 result: {010,000,110,100}
```

 Note that we can now read off the shared face relative to each Block
 easily:

```
Block1 result: Upper eta (All binary Block1 ids have a 1 in the second position)
Block2 result: Lower xi (All binary Block2 ids have a 0 in the third position)
```

Now we know that `Direction<3>::%upper_eta()`
in Block1 corresponds to `Direction<3>::%lower_xi().%opposite()` in Block2.

The use of `.%opposite()` is a result of the Directions to a Block face being
anti-parallel because each Block lies on the opposite side of the shared face.<br>

The remaining two correspondences are given by the Alignment of the shared
face. It is useful to know the following information:<br>
In the Local CNS, if an edge lies along the xi direction, if one takes the
two corners making up that edge and takes the difference of their ids, one
 always gets the result \f$ \pm 1\f$. Similarly, if the edge lies in the eta
 direction, the result will be \f$ \pm 2\f$. Finally, if the edge lies in the
 zeta direction, the result will be \f$ \pm 4\f$. We use this information to
 determine the Alignment of the shared face:

```
Block1: 3-2=1  => This edge is in the +xi direction.
Block2: 0-2=-2 => This edge is in the -eta direction.
Then, +xi in Block1 corresponds to -eta in Block2.

Block1: 6-2=4 => This edge is in the +zeta direction.
Block2: 6-2=4 => This edge is in the +zeta direction.
Then, +zeta in Block1 corresponds to +zeta in Block2.
```

The corresponding directions in each Block have now been deduced.

To confirm, we can use the other ids as well and arrive at the same result:<br>

```
Block1: 7-6-1  => +xi
Block2: 4-6=-2 => -eta

Block1: 7-3=4  => +zeta
Block2: 4-0=4  => +zeta
```

### Setting Periodic Boundary Conditions
It is also possible to identify faces of a Block using the subset CNS. For
example, to identify the lower zeta face with the upper zeta face of a Block
where the corners are labeled `{3,0,4,1,9,6,10,7}`, one may supply the lists
`{3,0,4,1}` and `{9,6,10,7}` to the `set_periodic_boundaries` function.
\note The `set_periodic_boundaries` function is sensitive to the order of the
corners in the lists supplied as arguments. This is because the function
identifies corners and edges with each other as opposed to simply faces. This
allows the user to specify more peculiar boundary conditions. For example,
using `{3,0,4,1}` and `{6,7,9,10}` to set the periodic boundaries will identify
the lower zeta face with the upper zeta face, but after a rotation of a
quarter-turn.

For reference, here are the corners to use for each face for a Block with
corners labelled as `{0,1,2,3,4,5,6,7}` to set up periodic boundary conditions
in each dimension, i.e. a \f$\mathrm{T}^3\f$ topology:

Face | Corners
------|--------
upper xi| `{1,3,5,7}`
lower xi| `{0,2,4,6}`
upper eta| `{2,3,6,7}`
lower eta| `{0,1,4,5}`
upper zeta| `{4,5,6,7}`
lower zeta| `{0,1,2,3}`


