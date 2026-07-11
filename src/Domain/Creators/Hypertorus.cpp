// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Hypertorus.hpp"

#include <array>
#include <memory>
#include <numbers>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Literals.hpp"

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {

template <size_t Dim>
Hypertorus<Dim>::Hypertorus(
    const std::array<double, Dim>& lower_bounds,
    const std::array<double, Dim>& upper_bounds,
    const std::array<size_t, Dim>& initial_max_modes,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
        time_dependence,
    const Options::Context& context)
    : lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      time_dependence_(std::move(time_dependence)) {
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(initial_num_points_, d) = 2 * gsl::at(initial_max_modes, d) + 1;
  }
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<Dim>>();
  }
  for (size_t d = 0; d < Dim; ++d) {
    if (gsl::at(lower_bounds_, d) >= gsl::at(upper_bounds_, d)) {
      PARSE_ERROR(context,
                  "Lower bound ("
                      << gsl::at(lower_bounds_, d)
                      << ") must be strictly smaller than upper bound ("
                      << gsl::at(upper_bounds_, d) << ") in dimension " << d
                      << ".");
    }
  }
}

template <size_t Dim>
Domain<Dim> Hypertorus<Dim>::create_domain() const {
  constexpr double two_pi = 2. * std::numbers::pi;
  auto block_map = [this]() {
    if constexpr (Dim == 1) {
      return Affine{0., two_pi, lower_bounds_[0], upper_bounds_[0]};
    } else if constexpr (Dim == 2) {
      return Affine2D{Affine{0., two_pi, lower_bounds_[0], upper_bounds_[0]},
                      Affine{0., two_pi, lower_bounds_[1], upper_bounds_[1]}};
    } else {
      return Affine3D{Affine{0., two_pi, lower_bounds_[0], upper_bounds_[0]},
                      Affine{0., two_pi, lower_bounds_[1], upper_bounds_[1]},
                      Affine{0., two_pi, lower_bounds_[2], upper_bounds_[2]}};
    }
  }();
  auto stationary_map =
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          std::move(block_map));

  std::vector<Block<Dim>> blocks;
  blocks.reserve(1);
  blocks.emplace_back(std::move(stationary_map), 0_st,
                      DirectionMap<Dim, BlockNeighbors<Dim>>{},
                      block_names_.at(0), domain::topologies::hypertorus<Dim>);

  Domain<Dim> domain{std::move(blocks), {}, block_groups()};

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps_grid_to_inertial(1)[0]),
        std::move(time_dependence_->block_maps_grid_to_distorted(1)[0]),
        std::move(time_dependence_->block_maps_distorted_to_inertial(1)[0]));
  }
  return domain;
}

template <size_t Dim>
std::vector<DirectionMap<
    Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
Hypertorus<Dim>::external_boundary_conditions() const {
  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  return boundary_conditions;
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>> Hypertorus<Dim>::initial_extents() const {
  return {initial_num_points_};
}

template <size_t Dim>
std::vector<std::array<size_t, Dim>>
Hypertorus<Dim>::initial_refinement_levels() const {
  return {make_array<Dim>(0_st)};
}

template <size_t Dim>
std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Hypertorus<Dim>::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependence_->functions_of_time(initial_expiration_times);
}

template class Hypertorus<1>;
template class Hypertorus<2>;
template class Hypertorus<3>;

}  // namespace domain::creators
