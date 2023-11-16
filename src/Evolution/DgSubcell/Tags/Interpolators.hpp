// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <optional>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
namespace intrp {
template <size_t Dim>
class Irregular;
}  // namespace intrp
/// \endcond

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
}  // namespace evolution::dg::subcell::Tags
