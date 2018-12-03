\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# Domain Concepts {#domain_concepts}

* Logical Coordinates:<br>
  The coordinate frame of a reference cell.

* Domain:<br>
  The computational domain is partitioned into cells called Blocks. Furthermore, each Block
  must have at most one neighboring Block in each Direction.

* Block:<br>
  A reference cell embedded into a subset of the computational domain. Blocks
  are identified by unique integral values.

* Orientation:<br>
  The information of how the Logical Coordinates between Blocks are related.

* \ref BlockNeighbor "Block Neighbor":<br>
  The identity and Orientation of a neighbor of a given Block.

* Direction:<br>
  A logical coordinate axis and a label "Upper" or "Lower" depending on whether
  the Direction is aligned with or anti-aligned with the logical coordinate
  axis, respectively.

* Refinement Level:<br>
  The number of times a Block is split in half in a given dimension.

* Segment:<br>
  The specific subset of the interval \f$[-1, 1]\f$ defined by the Refinement Level and an integer label
  such that the Segment's interval is \f$[-1 + 2*(N/D), -1 + 2*(N+1)/D]\f$ where N is the integer label and
  D is \f$2^{(\textrm{Refinement Level})}\f$.

* \ref SegmentId "Segment Identifier":<br>
  The Refinement Level and an integer labeling a Segment.

* Element:<br>
  A refined subregion of a Block defined by its Segments in each
  logical coordinate. The properties of the Element
  are inherited from the block, i.e. it is self-similar to the Block.

* \ref ElementId "Element Identifier":<br>
  The Block Identifier containing the Element and a Segment Identifier
  in each logical coordinate axis.

* Mesh:<br>
  A regular set of grid points associated with an Element. Represented by the data structure Mesh.

* External Boundary:<br>
  A face of a Block or Element that has no neighbor.

* Internal Boundary:<br>
  A boundary that is not an External Boundary.

* \ref Neighbors "Face Neighbors" (Rename C++ Neighbors class to FaceNeighbors):<br>
  The identities and Orientation of the neighbors in a particular direction of a given Element.

* External Boundary Condition:<br>
  A prescription for updating the solution on an External Boundary. Each
  External Boundary of Block has exactly one External Boundary Condition
  associated with it.
