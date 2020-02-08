// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Interval.hpp"

#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"          // IWYU pragma: keep
#include "Domain/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"

/// \cond
namespace Frame {
struct Inertial;
struct Logical;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
Interval::Interval(
    typename LowerBound::type lower_x, typename UpperBound::type upper_x,
    typename IsPeriodicIn::type is_periodic_in_x,
    typename InitialRefinement::type initial_refinement_level_x,
    typename InitialGridPoints::type initial_number_of_grid_points_in_x,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
        time_dependence) noexcept
    // clang-tidy: trivially copyable
    : lower_x_(std::move(lower_x)),                        // NOLINT
      upper_x_(std::move(upper_x)),                        // NOLINT
      is_periodic_in_x_(std::move(is_periodic_in_x)),      // NOLINT
      initial_refinement_level_x_(                         // NOLINT
          std::move(initial_refinement_level_x)),          // NOLINT
      initial_number_of_grid_points_in_x_(                 // NOLINT
          std::move(initial_number_of_grid_points_in_x)),  // NOLINT
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<1>>();
  }
}

Domain<1> Interval::create_domain() const noexcept {
  Domain<1> domain{
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Affine{-1., 1., lower_x_[0], upper_x_[0]}),
      std::vector<std::array<size_t, 2>>{{{1, 2}}},
      is_periodic_in_x_[0] ? std::vector<PairOfFaces>{{{1}, {2}}}
                           : std::vector<PairOfFaces>{}};
  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps(1)[0]));
  }
  return domain;
}

std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
Interval::TimeDependence::default_value() noexcept {
  return std::make_unique<domain::creators::time_dependence::None<1>>();
}

std::vector<std::array<size_t, 1>> Interval::initial_extents() const noexcept {
  return {{{initial_number_of_grid_points_in_x_}}};
}

std::vector<std::array<size_t, 1>> Interval::initial_refinement_levels() const
    noexcept {
  return {{{initial_refinement_level_x_}}};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Interval::functions_of_time() const noexcept {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time();
  }
}
}  // namespace creators
}  // namespace domain
