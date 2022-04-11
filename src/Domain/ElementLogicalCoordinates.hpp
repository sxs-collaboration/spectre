// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace domain {
class BlockId;
}  // namespace domain
class DataVector;
template <size_t VolumeDim>
class ElementId;
template <typename IdType, typename DataType>
class IdPair;
/// \endcond

/// \ingroup ComputationalDomainGroup
///
/// Holds element logical coordinates of an arbitrary set of points on
/// a single `Element`.  The arbitrary set of points is assumed to be
/// a subset of a larger set of points spanning multiple `Element`s,
/// and this class holds `offsets` that index into that larger set of
/// points.
///
/// \details `offsets.size()` is the same as the size of the `DataVector`
/// inside `element_logical_coords`.
///
/// This is used during the process of interpolating volume quantities
/// on the `Elements` (e.g. the spatial metric) onto an arbitrary set
/// of points (e.g. the points on an apparent horizon or a
/// wave-extraction surface) expressed in some frame.  Here is an
/// outline of how this interpolation proceeds, and where
/// `element_logical_coordinates` and `block_logical_coordinates` fit
/// into the picture:
///
/// Assume some component (e.g. HorizonA) has a `Tensor<DataVector>`
/// of target coordinate points in some coordinate frame.  The goal is
/// to determine the `Element` and logical coordinates of each point,
/// have each `Element` interpolate volume data onto the points
/// contained inside that `Element`, and send the interpolated data
/// back to the component.  The first step of this process is to
/// determine the block_id and block_logical_coordinates of each
/// point; this is done by the component (e.g. HorizonA), which calls
/// the function `block_logical_coordinates` on its full set of target
/// points.  The result of `block_logical_coordinates` is then
/// communicated to the members of a NodeGroup component
/// (e.g. HorizonManager).  Each node of the NodeGroup then calls
/// `element_logical_coordinates`, which returns a map of `ElementId`
/// to `ElementLogicalCoordHolder` for all the `Element`s on that node
/// that contain one or more of the target points. The NodeGroup
/// (which already has received the volume data from the `Elements` on
/// that node), interpolates the volume data to the element logical
/// coordinates for all of these `ElementId`s.  The `offsets` in the
/// `ElementLogicalCoordHolder` are the indices into the `DataVectors`
/// of the original target coordinates and will be used to assemble
/// the interpolated data into `Tensor<DataVector>`s that have the
/// same ordering as the original target coordinates. The NodeGroups
/// perform a reduction to get the data back to the original
/// component.
template <size_t Dim>
struct ElementLogicalCoordHolder {
  tnsr::I<DataVector, Dim, Frame::ElementLogical> element_logical_coords;
  std::vector<size_t> offsets;
};

/// \ingroup ComputationalDomainGroup
///
/// Given a set of points in block logical coordinates and their
/// `BlockIds`, as returned from the function
/// `block_logical_coordinates`, determines which `Element`s in a list
/// of `ElementId`s contains each point, and determines the element
/// logical coordinates of each point.
///
/// \details Returns a std::unordered_map from `ElementId`s to
/// `ElementLogicalCoordHolder`s.
/// It is expected that only a subset of the points will be found
/// in the given `Element`s.
/// Boundary points: If a point is on the boundary of an Element, it is
/// considered contained in that Element only if it is on the lower bound
/// of the Element, or if it is on the upper bound of the element and that
/// upper bound coincides with the upper bound of the containing Block.
/// This means that each boundary point is contained in one and only one
/// Element.  We assume that the input block_coord_holders associates
/// a point on a Block boundary with only a single Block, the one with
/// the smaller BlockId, which is always the lower-bounding Block.
///
/// \code
///  <---    Block 0   ---> <---   Block 1   --->
///  |          |          |          |          |
/// P_0   E0   P_1   E1   P_2   E2   P_3   E3   P_4
///  |          |          |          |          |
///
/// For example, the above 1D diagram shows four Elements labeled E0
/// through E3, and five boundary points labeled P_0 through P_4 (where
/// P_0 and P_4 are external boundaries).  There are two Blocks.  This
/// algorithm assigns each boundary point to one and only one Element as
/// follows:
/// P_0 -> E0
/// P_1 -> E1
/// P_2 -> E1 (Note: block_coord_holders includes P_2 only in Block 0)
/// P_3 -> E3
/// P_4 -> E3
/// \endcode
template <size_t Dim>
auto element_logical_coordinates(
    const std::vector<ElementId<Dim>>& element_ids,
    const std::vector<std::optional<IdPair<
        domain::BlockId, tnsr::I<double, Dim, typename Frame::BlockLogical>>>>&
        block_coord_holders)
    -> std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>>;
