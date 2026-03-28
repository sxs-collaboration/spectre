// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CartoonCylinder.hpp"

#include <array>
#include <memory>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {

CartoonCylinder::CartoonCylinder(
    const std::array<double, 2> lower_bounds,
    const std::array<double, 2> upper_bounds,
    const std::array<size_t, 2> initial_refinement_levels,
    const std::array<size_t, 2> initial_num_points,
    const std::array<CoordinateMaps::Distribution, 2> distributions,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        2>
        boundary_conditions,
    const Options::Context& context)
    : lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      initial_refinement_levels_(initial_refinement_levels),
      initial_num_points_(initial_num_points),
      distributions_(distributions),
      boundary_conditions_(std::move(boundary_conditions)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if (gsl::at(lower_bounds_, 0) < 0) {
    PARSE_ERROR(context,
                "The lower bound for the x dimension must be >= 0, but got "
                    << gsl::at(lower_bounds_, 0) << ".");
  }
  for (size_t d = 0; d < 2; ++d) {
    if (gsl::at(lower_bounds_, d) >= gsl::at(upper_bounds_, d)) {
      PARSE_ERROR(context,
                  "Lower bound ("
                      << gsl::at(lower_bounds_, d)
                      << ") must be strictly smaller than upper bound ("
                      << gsl::at(upper_bounds_, d) << ") in dimension " << d
                      << ".");
    }
  }
  for (size_t d = 0; d < 2; ++d) {
    const auto& [lower_bc, upper_bc] = gsl::at(boundary_conditions_, d);
    if (lower_bc == nullptr and upper_bc == nullptr) {
      PARSE_ERROR(context, "None of the boundary conditions can be nullptr");
    }
    using domain::BoundaryConditions::is_none;
    if (is_none(lower_bc) or is_none(upper_bc)) {
      PARSE_ERROR(
          context,
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that.");
    }
    using domain::BoundaryConditions::is_periodic;
    if (d == 0 and (is_periodic(lower_bc) or is_periodic(upper_bc))) {
      PARSE_ERROR(
          context,
          "Cannot have periodic boundary conditions in the x dimension.");
    }
    if (d == 1) {
      if (is_periodic(lower_bc) != is_periodic(upper_bc)) {
        PARSE_ERROR(context,
                    "Periodic boundary conditions must be applied for both "
                    "upper and lower directions in the y dimension or none.");
      }
      is_periodic_in_y_ = is_periodic(lower_bc);
    }
  }
  domain_ = build_domain(context);
}

Domain<3> CartoonCylinder::create_domain(
    const Options::Context& /*context*/) const {
  using Interval = CoordinateMaps::Interval;
  using Identity1D = CoordinateMaps::Identity<1>;
  using cartoon_cylinder_map =
      CoordinateMaps::ProductOf3Maps<Interval, Interval, Identity1D>;
  const Identity1D identity_map;

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  auto block_map = cartoon_cylinder_map{
      {-1.0, 1.0, lower_bounds_[0], upper_bounds_[0], distributions_[0], 0.0},
      {-1.0, 1.0, lower_bounds_[1], upper_bounds_[1], distributions_[1], 0.0},
      identity_map};
  auto stationary_map =
      make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(block_map);

  if (is_periodic_in_y_) {
    std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors;
    std::vector<PairOfFaces> identifications{};
    identifications.push_back({{0, 1, 4, 5}, {2, 3, 6, 7}});

    std::vector<std::array<size_t, two_to_the(3_st)>> block_corners{1};
    std::iota(block_corners[0].begin(), block_corners[0].end(), 0_st);

    set_internal_boundaries<3>(&neighbors, block_corners);
    set_identified_boundaries<3>(identifications, block_corners, &neighbors);
    blocks.emplace_back(std::move(stationary_map), 0, std::move(neighbors[0]),
                        block_names_.at(0),
                        domain::topologies::cartoon_cylinder);
  } else {
    const DirectionMap<3, BlockNeighbors<3>> neighbors;
    blocks.emplace_back(std::move(stationary_map), 0, neighbors,
                        block_names_.at(0),
                        domain::topologies::cartoon_cylinder);
  }

  Domain<3> domain(std::move(blocks), {}, block_groups());

  if (not time_dependence_->is_none()) {
    domain.inject_time_dependent_map_for_block(
        0, std::move(time_dependence_->block_maps_grid_to_inertial(1)[0]),
        std::move(time_dependence_->block_maps_grid_to_distorted(1)[0]),
        std::move(time_dependence_->block_maps_distorted_to_inertial(1)[0]));
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
CartoonCylinder::external_boundary_conditions() const {
  if (boundary_conditions_[0][0] == nullptr) {
#ifdef SPECTRE_DEBUG
    for (size_t d = 0; d < 2; ++d) {
      ASSERT(gsl::at(boundary_conditions_, d)[0] == nullptr and
                 gsl::at(boundary_conditions_, d)[1] == nullptr,
             "Boundary conditions must be set for all directions or none.");
    }
#endif  // SPECTRE_DEBUG
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{1};
  for (size_t d = 0; d < 2; ++d) {
    if (not is_periodic_in_y_ or d != 1) {
      const auto& [lower_bc, upper_bc] = gsl::at(boundary_conditions_, d);
      boundary_conditions[0][Direction<3>{d, Side::Lower}] =
          lower_bc->get_clone();
      boundary_conditions[0][Direction<3>{d, Side::Upper}] =
          upper_bc->get_clone();
    }
  }
  return boundary_conditions;
}

Domain<3> CartoonCylinder::create_domain() const { return domain_; }

std::vector<std::array<size_t, 3>> CartoonCylinder::initial_extents() const {
  // cartoon bases always have extents set to 1
  return {{initial_num_points_[0], initial_num_points_[1], 1}};
}

std::vector<std::array<size_t, 3>> CartoonCylinder::initial_refinement_levels()
    const {
  // cartoon bases always have refinement set to 0
  return {{initial_refinement_levels_[0], initial_refinement_levels_[1], 0_st}};
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CartoonCylinder::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
