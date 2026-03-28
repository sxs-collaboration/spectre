// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CartoonSphere1D.hpp"

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

namespace Frame {
struct BlockLogical;
struct Inertial;
}  // namespace Frame

namespace domain::creators {

CartoonSphere1D::CartoonSphere1D(
    double inner_bound, double outer_bound,
    typename InitialRadialRefinement::type&& initial_refinement_levels,
    typename InitialNumberOfRadialGridPoints::type&& initial_num_points,
    std::vector<double> radial_partitioning,
    const typename RadialDistributions::type& radial_distributions,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_bound_(inner_bound),
      outer_bound_(outer_bound),
      radial_partitioning_(std::move(radial_partitioning)),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }
  if (inner_bound_ >= outer_bound_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_bound_) + " and outer radius is " +
                    std::to_string(outer_bound_) + ".");
  }
  set_shell_distribution(make_not_null(&num_blocks_),
                         make_not_null(&radial_distributions_),
                         radial_partitioning_, radial_distributions,
                         inner_bound_, outer_bound_, "inner", "outer", context);
  const auto check_possible_list_instantiation =
      [this, &context](std::variant<size_t, std::vector<size_t>> input_param,
                       const std::string& name) {
        if (std::holds_alternative<size_t>(input_param)) {
          const auto input_value = std::get<size_t>(input_param);
          return std::vector<size_t>(num_blocks_, input_value);
        } else {
          const auto& input_vec = std::get<std::vector<size_t>>(input_param);
          if (input_vec.size() != num_blocks_) {
            PARSE_ERROR(
                context,
                name << " must be the same size as RadialDistributions (size="
                     << radial_distributions_.size()
                     << ") and one larger than RadialPartitioning (size="
                     << radial_partitioning_.size() << "), but has size "
                     << input_vec.size() << ".");
          }
          return input_vec;
        }
      };
  initial_refinement_levels_ = check_possible_list_instantiation(
      std::move(initial_refinement_levels), "InitialRadialRefinement");
  initial_num_points_ = check_possible_list_instantiation(
      std::move(initial_num_points), "InitialNumberOfRadialGridPoints");

  if ((inner_boundary_condition_ == nullptr) !=
      (outer_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_none;
  if (is_none(inner_boundary_condition_) or
      is_none(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with CartoonSphere1D");
  }
  for (size_t block = 0; block < num_blocks_; ++block) {
    const std::string block_name = "Block" + std::to_string(block);
    block_names_.emplace_back(block_name);
    block_groups_[block_name].insert(block_name);
  }
  domain_ = build_domain(context);
}

Domain<3> CartoonSphere1D::build_domain(
    const Options::Context& /*context*/) const {
  using Interval = CoordinateMaps::Interval;
  using Identity1D = CoordinateMaps::Identity<1>;
  using cartoon_sphere_map =
      CoordinateMaps::ProductOf3Maps<Interval, Identity1D, Identity1D>;
  const Identity1D identity_map;

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);
  const auto aligned = OrientationMap<3>::create_aligned();
  for (size_t i = 0; i < num_blocks_; ++i) {
    auto block_map = cartoon_sphere_map{
        {-1.0, 1.0, i == 0 ? inner_bound_ : radial_partitioning_[i - 1],
         i == num_blocks_ - 1 ? outer_bound_ : radial_partitioning_[i],
         radial_distributions_[i], 0.0},
        identity_map,
        identity_map};
    auto stationary_map =
        make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            block_map);
    DirectionMap<3, BlockNeighbors<3>> neighbors;
    if (i > 0) {
      neighbors.emplace(std::pair{Direction<3>::lower_xi(),
                                  BlockNeighbors<3>{i - 1, aligned}});
    }
    if (i < num_blocks_ - 1) {
      neighbors.emplace(std::pair{Direction<3>::upper_xi(),
                                  BlockNeighbors<3>{i + 1, aligned}});
    }
    blocks.emplace_back(std::move(stationary_map), i, std::move(neighbors),
                        block_names_.at(i), domain::topologies::cartoon_sphere);
  }

  Domain<3> domain(std::move(blocks), {}, block_groups_);

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
CartoonSphere1D::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};
  boundary_conditions[0][Direction<3>::lower_xi()] =
      inner_boundary_condition_->get_clone();
  boundary_conditions[num_blocks_ - 1][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

const Domain<3>& CartoonSphere1D::domain() const { return domain_; }

std::vector<std::array<size_t, 3>> CartoonSphere1D::initial_extents() const {
  std::vector<std::array<size_t, 3>> output;
  output.reserve(initial_num_points_.size());
  for (const auto& val : initial_num_points_) {
    // cartoon bases always have extents set to 1
    output.push_back({val, 1_st, 1_st});
  }
  return output;
}

std::vector<std::array<size_t, 3>> CartoonSphere1D::initial_refinement_levels()
    const {
  std::vector<std::array<size_t, 3>> output;
  output.reserve(initial_refinement_levels_.size());
  for (const auto& val : initial_refinement_levels_) {
    // cartoon bases always have no refinement
    output.push_back({val, 0_st, 0_st});
  }
  return output;
}

std::vector<std::string> CartoonSphere1D::block_names() const {
  return block_names_;
}

std::unordered_map<std::string, std::unordered_set<std::string>>
CartoonSphere1D::block_groups() const {
  return block_groups_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CartoonSphere1D::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  if (time_dependence_->is_none()) {
    return {};
  } else {
    return time_dependence_->functions_of_time(initial_expiration_times);
  }
}
}  // namespace domain::creators
