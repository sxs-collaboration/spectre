// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AlignedLattice.hpp"

#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace domain {
namespace creators {

template <size_t VolumeDim, typename TargetFrame>
AlignedLattice<VolumeDim, TargetFrame>::AlignedLattice(
    const typename BlockBounds::type block_bounds,
    const typename IsPeriodicIn::type is_periodic_in,
    const typename InitialRefinement::type initial_refinement_levels,
    const typename InitialGridPoints::type initial_number_of_grid_points,
    typename BlocksToExclude::type blocks_to_exclude) noexcept
    // clang-tidy: trivially copyable
    : block_bounds_(std::move(block_bounds)),         // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),     // NOLINT
      initial_refinement_levels_(                     // NOLINT
          std::move(initial_refinement_levels)),      // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      blocks_to_exclude_(std::move(blocks_to_exclude)),
      number_of_blocks_by_dim_{
          map_array(block_bounds_, [](const std::vector<double>& v) noexcept {
            return v.size() - 1;
          })} {
  if (not blocks_to_exclude_.empty() and
      alg::any_of(is_periodic_in_, [](const bool t) noexcept { return t; })) {
    ERROR(
        "Cannot exclude blocks as well as have periodic boundary "
        "conditions!");
  }
}

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>
AlignedLattice<VolumeDim, TargetFrame>::create_domain() const noexcept {
  if (blocks_to_exclude_.empty()) {
    return rectilinear_domain<VolumeDim, TargetFrame>(
        number_of_blocks_by_dim_, block_bounds_, {}, {}, is_periodic_in_);
  }
  return rectilinear_domain<VolumeDim, TargetFrame>(
      number_of_blocks_by_dim_, block_bounds_,
      {std::vector<Index<VolumeDim>>(blocks_to_exclude_.begin(),
                                     blocks_to_exclude_.end())},
      {}, make_array<VolumeDim>(false));
}

template <size_t VolumeDim, typename TargetFrame>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim, TargetFrame>::initial_extents() const noexcept {
  return {number_of_blocks_by_dim_.product() - blocks_to_exclude_.size(),
          initial_number_of_grid_points_};
}

template <size_t VolumeDim, typename TargetFrame>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim, TargetFrame>::initial_refinement_levels() const
    noexcept {
  return {number_of_blocks_by_dim_.product() - blocks_to_exclude_.size(),
          initial_refinement_levels_};
}

template class AlignedLattice<1, Frame::Inertial>;
template class AlignedLattice<1, Frame::Grid>;
template class AlignedLattice<2, Frame::Inertial>;
template class AlignedLattice<2, Frame::Grid>;
template class AlignedLattice<3, Frame::Inertial>;
template class AlignedLattice<3, Frame::Grid>;
}  // namespace creators
}  // namespace domain
