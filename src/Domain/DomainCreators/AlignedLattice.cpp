// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/DomainCreators/AlignedLattice.hpp"

#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame

namespace DomainCreators {

template <size_t VolumeDim, typename TargetFrame>
AlignedLattice<VolumeDim, TargetFrame>::AlignedLattice(
    const typename BlockBounds::type block_bounds,
    const typename IsPeriodicIn::type is_periodic_in,
    const typename InitialRefinement::type initial_refinement_levels,
    const typename InitialGridPoints::type
        initial_number_of_grid_points) noexcept
    // clang-tidy: trivially copyable
    : block_bounds_(std::move(block_bounds)),         // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),     // NOLINT
      initial_refinement_levels_(                     // NOLINT
          std::move(initial_refinement_levels)),      // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      number_of_blocks_by_dim_{
          map_array(block_bounds_, [](const std::vector<double>& v) noexcept {
            return v.size() - 1;
          })} {}

template <size_t VolumeDim, typename TargetFrame>
Domain<VolumeDim, TargetFrame>
AlignedLattice<VolumeDim, TargetFrame>::create_domain() const noexcept {
  return rectilinear_domain<VolumeDim, TargetFrame>(
      number_of_blocks_by_dim_, block_bounds_, {}, {}, is_periodic_in_);
}

template <size_t VolumeDim, typename TargetFrame>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim, TargetFrame>::initial_extents() const noexcept {
  return {number_of_blocks_by_dim_.product(), initial_number_of_grid_points_};
}

template <size_t VolumeDim, typename TargetFrame>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim, TargetFrame>::initial_refinement_levels() const
    noexcept {
  return {number_of_blocks_by_dim_.product(), initial_refinement_levels_};
}
}  // namespace DomainCreators

template class DomainCreators::AlignedLattice<1, Frame::Inertial>;
template class DomainCreators::AlignedLattice<1, Frame::Grid>;
template class DomainCreators::AlignedLattice<2, Frame::Inertial>;
template class DomainCreators::AlignedLattice<2, Frame::Grid>;
template class DomainCreators::AlignedLattice<3, Frame::Inertial>;
template class DomainCreators::AlignedLattice<3, Frame::Grid>;
