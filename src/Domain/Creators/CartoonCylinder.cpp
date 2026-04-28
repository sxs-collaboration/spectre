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
#include "Utilities/Gsl.hpp"

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {

// Does not take cartoon BC
detail::CartoonCylinderOptionsHelper::CartoonCylinderOptionsHelper(
    std::array<double, 2> lower_bounds, std::array<double, 2> upper_bounds,
    std::array<size_t, 2> initial_refinement_levels,
    std::array<size_t, 2> initial_num_points,
    std::array<CoordinateMaps::Distribution, 2> distributions,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        2>
        boundary_conditions,
    Options::Context context)
    : lower_bounds_(std::move(lower_bounds)),
      upper_bounds_(std::move(upper_bounds)),
      initial_refinement_levels_(std::move(initial_refinement_levels)),
      initial_num_points_(std::move(initial_num_points)),
      distributions_(std::move(distributions)),
      time_dependence_(std::move(time_dependence)),
      boundary_conditions_(std::move(boundary_conditions)),
      context_(std::move(context)) {}

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
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cartoon_boundary_condition,
    const Options::Context& context)
    : lower_bounds_(lower_bounds),
      upper_bounds_(upper_bounds),
      initial_refinement_levels_(initial_refinement_levels),
      initial_num_points_(initial_num_points),
      distributions_(distributions),
      boundary_conditions_(std::move(boundary_conditions)),
      cartoon_boundary_condition_(std::move(cartoon_boundary_condition)),
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
  if (cartoon_boundary_condition_ == nullptr) {
    PARSE_ERROR(
        context,
        "CartoonCylinder should only be used with systems that have a "
        "cartoon-style boundary condition, but none was provided. This "
        "means the system is not set up to use cartoon methods. Make sure your "
        "system has a boundary condition that inherits from MarkAsCartoon in "
        "its standard_boundary_conditions list.");
  }

  // Check if user mistakenly specified a cartoon BC as an external boundary
  bool found_cartoon_external = false;
  for (size_t d = 0; d < 2; ++d) {
    for (size_t side = 0; side < 2; ++side) {
      if (gsl::at(gsl::at(boundary_conditions_, d), side) != nullptr and
          domain::BoundaryConditions::is_cartoon(
              gsl::at(gsl::at(boundary_conditions_, d), side))) {
        found_cartoon_external = true;
        break;
      }
    }
    if (found_cartoon_external) {
      break;
    }
  }
  if (found_cartoon_external) {
    PARSE_ERROR(
        context,
        "Cartoon boundary conditions should not be specified as external "
        "boundary conditions. They are automatically applied to internal "
        "cartoon boundaries. Please choose different boundary conditions "
        "for the external boundaries.");
  }

  using_zernike_ = gsl::at(lower_bounds_, 0) == 0.0;
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
  num_blocks_ = 1;
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
  block_names_.emplace_back("Block0");
  block_groups_["CartoonCylinder"].insert("Block0");
}

Domain<3> CartoonCylinder::create_domain() const {
  using Interval = CoordinateMaps::Interval;
  using Identity1D = CoordinateMaps::Identity<1>;
  using cartoon_cylinder_map =
      CoordinateMaps::ProductOf3Maps<Interval, Interval, Identity1D>;
  const Identity1D identity_map;

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  const auto topology = using_zernike_
                            ? domain::topologies::cartoon_cylinder_inner
                            : domain::topologies::cartoon_cylinder;
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
                        block_names_.at(0), topology);
  } else {
    const DirectionMap<3, BlockNeighbors<3>> neighbors;
    blocks.emplace_back(std::move(stationary_map), 0, neighbors,
                        block_names_.at(0), topology);
  }

  Domain<3> domain(std::move(blocks), {}, block_groups());

  if (not time_dependence_->is_none()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial =
            time_dependence_->block_maps_grid_to_inertial(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted =
            time_dependence_->block_maps_grid_to_distorted(num_blocks_);
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial =
            time_dependence_->block_maps_distorted_to_inertial(num_blocks_);
    for (size_t block_id = 0; block_id < num_blocks_; ++block_id) {
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(block_maps_grid_to_inertial[block_id]),
          std::move(block_maps_grid_to_distorted[block_id]),
          std::move(block_maps_distorted_to_inertial[block_id]));
    }
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
      boundary_conditions{num_blocks_};
  const auto& [lower_x_bc, upper_x_bc] = gsl::at(boundary_conditions_, 0);
  if (using_zernike_) {
    boundary_conditions[0][Direction<3>::lower_xi()] =
        cartoon_boundary_condition_->get_clone();
  } else {
    boundary_conditions[0][Direction<3>::lower_xi()] = lower_x_bc->get_clone();
  }
  boundary_conditions[num_blocks_ - 1][Direction<3>::upper_xi()] =
      upper_x_bc->get_clone();

  if (not is_periodic_in_y_) {
    const auto& [lower_y_bc, upper_y_bc] = gsl::at(boundary_conditions_, 1);
    boundary_conditions[0][Direction<3>::lower_eta()] = lower_y_bc->get_clone();
    boundary_conditions[0][Direction<3>::upper_eta()] = upper_y_bc->get_clone();
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> CartoonCylinder::initial_extents() const {
  // cartoon bases always have extents set to 1
  return std::vector<std::array<size_t, 3>>{
      {initial_num_points_[0], initial_num_points_[1], 1}};
}

std::vector<std::array<size_t, 3>> CartoonCylinder::initial_refinement_levels()
    const {
  // cartoon bases always have refinement set to 0
  return std::vector<std::array<size_t, 3>>{
      {initial_refinement_levels_[0], initial_refinement_levels_[1], 0_st}};
}

std::vector<std::string> CartoonCylinder::block_names() const {
  return block_names_;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
CartoonCylinder::block_groups() const {
  return block_groups_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CartoonCylinder::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependence_->functions_of_time(initial_expiration_times);
}
}  // namespace domain::creators
