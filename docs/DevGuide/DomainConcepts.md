\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Domain Concepts {#domain_concepts}

* Computational Domain:<br>
  The region of spacetime on which a numerical simulation is performed.

* Inertial Coordinate Frame:<br>
  A global coordinate frame covering the computational domain that is also the
  coordinate frame in which the initial (boundary) value problem that is being
  solved is specified.  Denoted by Frame::Inertial.

* Logical Coordinate Frame:<br> The coordinate frame of a reference
  cell.  In the cell the logical coordinates cover the interval
  \f$[-1, 1]\f$ in each dimension.  Currently the logical coordinates
  are Cartesian.

* CoordinateMap "Coordinate Map":<br>
  A mapping between two coordinate frames.  Coordinate maps are allowed to be
  time-dependent, as long as the time coordinate itself is unchanged.

* Direction:<br>
  A logical coordinate axis and a label "Upper" or "Lower" depending on whether
  the Direction is aligned with or anti-aligned with the logical coordinate
  axis, respectively.

* Block:<br>
  The computational domain is partitioned into a set of non-overlapping,
  distorted reference cells called Blocks. Each Block must have at most one
  neighboring Block in each Direction.  The reference cell is embedded into a
  subset of the computational domain using a Coordinate Map from the
  logical frame of the Block to the global inertial frame.  Blocks are
  identified by unique integral values.

* Block Logical Coordinate Frame:<br>
  The logical coordinate frame of a Block, denoted by Frame::BlockLogical.
  The only requirement upon the logical coordinate frames of neighboring Blocks
  is that they have the same Coordinate Map from their logical coordinate frame
  to the global inertial frame on their shared boundary up to a mapping that
  swaps or negates the logical coordinate axes.  In other words, at each point
  on their shared boundary, the logical coordinates of the two blocks are
  equal after possibly applying a permutation and sign flips.

* Grid Coordinate Frame:<br>
  For time-dependent coordinate maps, it is useful (e.g. for computational
  efficiency) to split the Coordinate Map from Frame::BlockLogical to
  Frame::Inertial into a composition of two maps.  This is done by introducing
  an intermediate coordinate frame (denoted by Frame::Grid) such that the
  mapping from Frame::BlockLogical to Frame::Grid is time-independent, and
  the mapping from Frame::Grid to Frame::Inertial is time-dependent.

* Orientation:<br>
  The information of how the Block Logical Coordinates of neighboring Blocks are
  related.

* \ref BlockNeighbor "Block Neighbor":<br>
  The identity and Orientation of a neighboring Block of a given Block.

* Element:<br> A reference cell that is a refined subregion of a Block
  defined by its Segments in each dimension. The properties of the
  Element (e.g coordinate map) are inherited from its Block, i.e. it
  is self-similar to the Block.

* Refinement Level:<br>
  The number of times a Block is split in half in a given dimension.

* Segment:<br> In each dimension, the specific subset of the block
  logical coordinate interval \f$[-1, 1]\f$ defined by the Refinement
  Level and an integer label such that the Segment's interval is
  \f$[-1 + 2 \frac{i}{N}, -1 + 2 \frac{i+1}{N}]\f$ where \f$i\f$ is
  the integer label and \f$N = 2^{(\textrm{Refinement Level})}\f$ is
  the number of segments on the Refinement Level.

* Element Logical Coordinate Frame:<br>
  The logical coordinate frame of an Element.  In each dimension, the Element
  Logical Coordinates are related to the Block Logical Coordinates by an affine
  mapping of the interval \f$[-1, 1]\f$ to the interval covered by the Segment
  in that dimension.

* \ref SegmentId "Segment Identifier":<br>
  The Refinement Level and an integer labeling a Segment.

* \ref ElementId "Element Identifier":<br>
  The Block Identifier containing the Element and a Segment Identifier
  in each dimension.

* External Boundary:<br>
  A face of a Block or Element that has no neighbor.

* Internal Boundary:<br>
  A boundary that is not an External Boundary.

* Neighbors:<br>
  The identities and Orientation of the neighboring Elements of a given Element
  in a particular Direction .

* External Boundary Condition:<br>
  A prescription for updating the solution on an External Boundary. Each
  External Boundary of a Block has exactly one External Boundary Condition
  associated with it.
