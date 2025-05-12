// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
namespace PUP {
class er;
}  // namespace PUP
namespace intrp {
template <size_t Dim>
class Irregular;
}  // namespace intrp
/// \endcond

namespace interpolators_detail {
template <size_t Dim>
struct ExtensionDirection {
  static constexpr size_t dim = Dim;

  // Stores the direction along which a neighbor's ghost data lies
  // outside of the local element's domain and require extension.
  // This is used to determine the direction in which to extend the
  // local element's domain to include the neighbor's ghost data.
  Direction<Dim> direction_to_extend;

  ExtensionDirection() = default;
  explicit ExtensionDirection(Direction<Dim> direction_to_extend_v)
      : direction_to_extend(direction_to_extend_v) {
    ASSERT(
        direction_to_extend.dimension() < dim,
        "Invalid direction: dimension must be less than the volume dimension.");
  }

  // Serialization for Charm++
  void pup(PUP::er& p) { p | direction_to_extend; };
};
}  // namespace interpolators_detail

namespace evolution::dg::subcell::Tags {
/*!
 * \brief An `intrp::Irregular` from our FD grid to our neighbors' FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromFdToNeighborFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief An `intrp::Irregular` from our DG grid to our neighbors' FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromDgToNeighborFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief An `intrp::Irregular` from our neighbors' DG grid to our FD grid.
 *
 * Values are only set if the neighboring elements' logical coordinate axes
 * are not aligned with ours. I.e., on block boundaries.
 */
template <size_t Dim>
struct InterpolatorsFromNeighborDgToFd : db::SimpleTag {
  using type = DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>;
};

/*!
 * \brief Records directions in which neighbor ghost zones fall outside
 * the local element's domain and require extension.
 *
 * This tag is used on block boundaries where logical coordinate axes of
 * neighboring elements are not aligned, and ghost data cannot be
 * directly exchanged.
 */
template <size_t Dim>
struct ExtensionDirections : db::SimpleTag {
  using type = DirectionMap<Dim, interpolators_detail::ExtensionDirection<Dim>>;
};
}  // namespace evolution::dg::subcell::Tags
