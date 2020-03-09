// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AlignedLattice.hpp"

#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace domain {
namespace creators {

using ::operator<<;
template <size_t VolumeDim>
std::ostream& operator<<(
    std::ostream& /*s*/,
    const RefinementRegion<VolumeDim>& /*unused*/) noexcept {
  ERROR(
      "RefinementRegion stream operator is only for option parsing and "
      "should never be called.");
}

template <size_t VolumeDim>
AlignedLattice<VolumeDim>::AlignedLattice(
    const typename BlockBounds::type block_bounds,
    const typename IsPeriodicIn::type is_periodic_in,
    const typename InitialRefinement::type initial_refinement_levels,
    const typename InitialGridPoints::type initial_number_of_grid_points,
    typename RefinedGridPoints::type refined_grid_points,
    typename BlocksToExclude::type blocks_to_exclude) noexcept
    // clang-tidy: trivially copyable
    : block_bounds_(std::move(block_bounds)),         // NOLINT
      is_periodic_in_(std::move(is_periodic_in)),     // NOLINT
      initial_refinement_levels_(                     // NOLINT
          std::move(initial_refinement_levels)),      // NOLINT
      initial_number_of_grid_points_(                 // NOLINT
          std::move(initial_number_of_grid_points)),  // NOLINT
      refined_grid_points_(std::move(refined_grid_points)),
      blocks_to_exclude_(std::move(blocks_to_exclude)),
      number_of_blocks_by_dim_{map_array(
          block_bounds_,
          [](const std::vector<double>& v) noexcept { return v.size() - 1; })} {
  if (not blocks_to_exclude_.empty() and
      alg::any_of(is_periodic_in_, [](const bool t) noexcept { return t; })) {
    ERROR(
        "Cannot exclude blocks as well as have periodic boundary "
        "conditions!");
  }
  for (const auto& refinement_region : refined_grid_points_) {
    for (size_t i = 0; i < VolumeDim; ++i) {
      if (gsl::at(refinement_region.upper_corner_index, i) >=
          gsl::at(block_bounds_, i).size()) {
        ERROR("Refinement region extends to "
              << refinement_region.upper_corner_index
              << ", which is outside the domain");
      }
    }
  }
}

template <size_t VolumeDim>
Domain<VolumeDim> AlignedLattice<VolumeDim>::create_domain() const noexcept {
  if (blocks_to_exclude_.empty()) {
    return rectilinear_domain<VolumeDim>(
        number_of_blocks_by_dim_, block_bounds_, {}, {}, is_periodic_in_);
  }
  return rectilinear_domain<VolumeDim>(
      number_of_blocks_by_dim_, block_bounds_,
      {std::vector<Index<VolumeDim>>(blocks_to_exclude_.begin(),
                                     blocks_to_exclude_.end())},
      {}, make_array<VolumeDim>(false));
}

template <size_t VolumeDim>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim>::initial_extents() const noexcept {
  std::vector<std::array<size_t, VolumeDim>> all_extents;
  for (const auto& block_index : indices_for_rectilinear_domains(
           number_of_blocks_by_dim_,
           std::vector<Index<VolumeDim>>(blocks_to_exclude_.begin(),
                                         blocks_to_exclude_.end()))) {
    std::array<size_t, VolumeDim> extents = initial_number_of_grid_points_;
    for (const auto& refinement_region : refined_grid_points_) {
      for (size_t d = 0; d < VolumeDim; ++d) {
        if (block_index[d] < gsl::at(refinement_region.lower_corner_index, d) or
            block_index[d] >=
                gsl::at(refinement_region.upper_corner_index, d)) {
          goto next_region;
        }
      }
      extents = refinement_region.refinement;
    next_region:;
    }
    all_extents.push_back(extents);
  }
  return all_extents;
}

template <size_t VolumeDim>
std::vector<std::array<size_t, VolumeDim>>
AlignedLattice<VolumeDim>::initial_refinement_levels() const noexcept {
  return {number_of_blocks_by_dim_.product() - blocks_to_exclude_.size(),
          initial_refinement_levels_};
}

template class AlignedLattice<1>;
template class AlignedLattice<2>;
template class AlignedLattice<3>;
template std::ostream& operator<<(
    std::ostream& /*s*/, const RefinementRegion<1>& /*unused*/) noexcept;
template std::ostream& operator<<(
    std::ostream& /*s*/, const RefinementRegion<2>& /*unused*/) noexcept;
template std::ostream& operator<<(
    std::ostream& /*s*/, const RefinementRegion<3>& /*unused*/) noexcept;
}  // namespace creators
}  // namespace domain
