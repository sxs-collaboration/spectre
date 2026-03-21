// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/AngularCylinder.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {
AngularCylinder::AngularCylinder(
    typename OuterRadius::type outer_radius,
    typename LowerZBound::type lower_z_bound,
    typename UpperZBound::type upper_z_bound,
    typename RadialPartitioning::type radial_partitioning,
    typename PartitioningInZ::type partitioning_in_z,
    typename InitialCylinderThetaGridPoints::type
        initial_cylinder_theta_grid_points,
    typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
    typename InitialHollowCylinderGridPoints::type
        initial_hollow_cylinder_grid_points,
    typename DistributionInZ::type distribution_in_z,
    typename InitialRefinementInZ::type initial_refinement_in_z,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    bool is_periodic_in_z, const Options::Context& context)
    : outer_radius_(outer_radius),
      lower_z_bound_(lower_z_bound),
      upper_z_bound_(upper_z_bound),
      is_periodic_in_z_(is_periodic_in_z),
      radial_partitioning_(std::move(radial_partitioning)),
      partitioning_in_z_(std::move(partitioning_in_z)),
      initial_cylinder_theta_grid_points_(initial_cylinder_theta_grid_points),
      initial_cylinder_z_grid_points_(initial_cylinder_z_grid_points),
      distribution_in_z_(std::move(distribution_in_z)),
      time_dependence_(std::move(time_dependence)) {
  if (time_dependence_ == nullptr) {
    time_dependence_ =
        std::make_unique<domain::creators::time_dependence::None<3>>();
  }

  if (outer_radius_ <= 0.0) {
    PARSE_ERROR(context,
                "OuterRadius must be positive, but is: " << outer_radius_);
  }
  if (lower_z_bound_ >= upper_z_bound_) {
    PARSE_ERROR(context, "LowerZBound must be less than UpperZBound, but lower="
                             << lower_z_bound_
                             << " and upper=" << upper_z_bound_);
  }

  if (not radial_partitioning_.empty()) {
    if (not std::is_sorted(radial_partitioning_.begin(),
                           radial_partitioning_.end())) {
      PARSE_ERROR(context,
                  "You must specify radial partitioning in ascending order.");
    }
    if (radial_partitioning_.front() <= 0) {
      PARSE_ERROR(context, "Radial partitions must be positive, got "
                               << radial_partitioning_.front());
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(context,
                  "Last radial partition must be smaller than the outer "
                  "radius, but is "
                      << radial_partitioning_.back() << " with outer radius "
                      << outer_radius_);
    }
    const auto duplicate = std::adjacent_find(radial_partitioning_.begin(),
                                              radial_partitioning_.end());
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }

  if (not partitioning_in_z_.empty()) {
    if (not std::is_sorted(partitioning_in_z_.begin(),
                           partitioning_in_z_.end())) {
      PARSE_ERROR(context, "Specify partitioning in z in ascending order.");
    }
    if (partitioning_in_z_.front() <= lower_z_bound_) {
      PARSE_ERROR(context,
                  "First z partition must be larger than LowerZBound, but is: "
                      << partitioning_in_z_.front());
    }
    if (partitioning_in_z_.back() >= upper_z_bound_) {
      PARSE_ERROR(context,
                  "Last z partition must be smaller than UpperZBound, but is: "
                      << partitioning_in_z_.back());
    }
  }

  const size_t num_radial_blocks = 1 + radial_partitioning_.size();
  const size_t num_layers = 1 + partitioning_in_z_.size();

  if (distribution_in_z_.size() != num_layers) {
    PARSE_ERROR(context,
                "Specify a 'DistributionInZ' for every layer. You specified "
                    << distribution_in_z_.size()
                    << " items, but the domain has " << num_layers
                    << " layers.");
  }
  if (distribution_in_z_.front() !=
      domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'DistributionInZ' must be 'Linear' for the lowermost "
                "layer because a 'Logarithmic' distribution places its "
                "singularity at 'LowerZBound'. Add entries to "
                "'PartitioningInZ' to add layers for which you can select "
                "different distributions along z.");
  }

  if (initial_cylinder_theta_grid_points_ % 2 != 1) {
    PARSE_ERROR(context,
                "The number of angular grid points must be odd (this helps "
                "with numerical stability), for center cylinder got "
                    << initial_cylinder_theta_grid_points_);
  }

  num_blocks_ = num_radial_blocks * num_layers;

  // Build block names and groups.
  // Block ordering: for each layer (outer loop), for each radial block (inner):
  //   block_id = layer * num_radial_blocks + radial_block
  block_names_.reserve(num_blocks_);
  for (size_t layer = 0; layer < num_layers; ++layer) {
    const std::string layer_prefix =
        num_layers > 1 ? "Layer" + std::to_string(layer) : "";
    const std::string center_name = layer_prefix + "CenterDisk";
    block_names_.emplace_back(center_name);
    if (num_layers > 1) {
      block_groups_[layer_prefix].insert(center_name);
    }
    for (size_t shell = 0; shell < radial_partitioning_.size(); ++shell) {
      const std::string shell_name =
          layer_prefix + "Shell" + std::to_string(shell);
      block_names_.emplace_back(shell_name);
      if (num_layers > 1) {
        block_groups_[layer_prefix].insert(shell_name);
      }
      block_groups_["Shells"].insert(shell_name);
    }
  }
  // InnerDisk holds the center disk blocks across layers
  for (size_t layer = 0; layer < num_layers; ++layer) {
    block_groups_["InnerDisks"].insert(block_names_[layer * num_radial_blocks]);
  }

  // Expand hollow cylinder grid points
  const size_t num_shells = radial_partitioning_.size();
  if (std::holds_alternative<std::array<size_t, 3>>(
          initial_hollow_cylinder_grid_points)) {
    const auto input_value =
        std::get<std::array<size_t, 3>>(initial_hollow_cylinder_grid_points);
    if (input_value[1] % 2 != 1) {
      PARSE_ERROR(context,
                  "The number of angular grid points must be odd (this helps "
                  "with numerical stability), for shell(s) got "
                      << input_value[1]);
    }
    initial_hollow_cylinder_grid_points_ =
        std::vector<std::array<size_t, 3>>(num_shells, input_value);
  } else {
    initial_hollow_cylinder_grid_points_ =
        std::get<std::vector<std::array<size_t, 3>>>(
            initial_hollow_cylinder_grid_points);
    if (initial_hollow_cylinder_grid_points_.size() != num_shells) {
      PARSE_ERROR(context,
                  "InitialHollowCylinderGridPoints must have one entry per "
                  "shell (RadialPartitioning size="
                      << num_shells << "), but has size "
                      << initial_hollow_cylinder_grid_points_.size() << ".");
    }
    for (size_t i = 0; i < initial_hollow_cylinder_grid_points_.size(); ++i) {
      if (initial_hollow_cylinder_grid_points_[i][1] % 2 != 1) {
        PARSE_ERROR(context,
                    "The number of angular grid points must be odd (this helps "
                    "with numerical stability), for shell "
                        << i << " got "
                        << initial_hollow_cylinder_grid_points_[i][1]);
      }
    }
  }

  // Expand initial refinement in z
  if (std::holds_alternative<size_t>(initial_refinement_in_z)) {
    const size_t single_refinement = std::get<size_t>(initial_refinement_in_z);
    initial_refinement_in_z_ =
        std::vector<size_t>(num_layers, single_refinement);
  } else {
    initial_refinement_in_z_ =
        std::get<std::vector<size_t>>(initial_refinement_in_z);
    if (initial_refinement_in_z_.size() != num_layers) {
      PARSE_ERROR(context,
                  "InitialRefinementInZ must have one entry per layer "
                  "(PartitioningInZ size + 1 = "
                      << num_layers << "), but has size "
                      << initial_refinement_in_z_.size() << ".");
    }
  }
}

AngularCylinder::AngularCylinder(
    typename OuterRadius::type outer_radius,
    typename LowerZBound::type lower_z_bound,
    typename UpperZBound::type upper_z_bound,
    typename RadialPartitioning::type radial_partitioning,
    typename PartitioningInZ::type partitioning_in_z,
    typename InitialCylinderThetaGridPoints::type
        initial_cylinder_theta_grid_points,
    typename InitialCylinderZGridPoints::type initial_cylinder_z_grid_points,
    typename InitialHollowCylinderGridPoints::type
        initial_hollow_cylinder_grid_points,
    typename DistributionInZ::type distribution_in_z,
    typename InitialRefinementInZ::type initial_refinement_in_z,
    std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
        time_dependence,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        lower_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        upper_z_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        mantle_boundary_condition,
    const Options::Context& context)
    : AngularCylinder(
          outer_radius, lower_z_bound, upper_z_bound,
          std::move(radial_partitioning), std::move(partitioning_in_z),
          initial_cylinder_theta_grid_points, initial_cylinder_z_grid_points,
          std::move(initial_hollow_cylinder_grid_points),
          std::move(distribution_in_z), std::move(initial_refinement_in_z),
          std::move(time_dependence),
          false,  // is_periodic_in_z
          context) {
  // NOLINTNEXTLINE
  lower_z_boundary_condition_ = std::move(lower_z_boundary_condition);
  // NOLINTNEXTLINE
  upper_z_boundary_condition_ = std::move(upper_z_boundary_condition);
  // NOLINTNEXTLINE
  mantle_boundary_condition_ = std::move(mantle_boundary_condition);

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  using domain::BoundaryConditions::is_periodic;
  if (lower_z_boundary_condition_ != nullptr) {
    if (is_none(lower_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for LowerZ. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(lower_z_boundary_condition_) xor
        is_periodic(upper_z_boundary_condition_)) {
      PARSE_ERROR(context,
                  "Either both lower and upper z-boundary conditions must "
                  "be periodic, or neither.");
    }
    if (is_periodic(lower_z_boundary_condition_) and
        is_periodic(upper_z_boundary_condition_)) {
      is_periodic_in_z_ = true;
      lower_z_boundary_condition_ = nullptr;
      upper_z_boundary_condition_ = nullptr;
    }
  }
  if (upper_z_boundary_condition_ != nullptr and
      is_none(upper_z_boundary_condition_)) {
    PARSE_ERROR(context,
                "None boundary condition is not supported for UpperZ. "
                "Use an outflow-type boundary condition instead.");
  }
  if (mantle_boundary_condition_ != nullptr) {
    if (is_none(mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "None boundary condition is not supported for Mantle. "
                  "Use an outflow-type boundary condition instead.");
    }
    if (is_periodic(mantle_boundary_condition_)) {
      PARSE_ERROR(context,
                  "A cylinder can't have periodic boundary conditions in "
                  "the radial direction.");
    }
  } else {
    if (lower_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but lower is. This "
                  "is probably a mistake");
    }
    if (upper_z_boundary_condition_ != nullptr) {
      PARSE_ERROR(context,
                  "Mantle boundary condition is not set, but upper z is. This "
                  "is probably a mistake");
    }
  }
}

Domain<3> AngularCylinder::create_domain() const {
  using Affine = CoordinateMaps::Affine;
  using Identity1D = CoordinateMaps::Identity<1>;
  using Interval = CoordinateMaps::Interval;
  const auto aligned = OrientationMap<3>::create_aligned();

  const size_t num_radial_blocks = 1 + radial_partitioning_.size();
  const size_t num_layers = 1 + partitioning_in_z_.size();

  // Build z-layer bounds
  std::vector<double> z_bounds;
  z_bounds.reserve(num_layers + 1);
  z_bounds.push_back(lower_z_bound_);
  for (const double z : partitioning_in_z_) {
    z_bounds.push_back(z);
  }
  z_bounds.push_back(upper_z_bound_);

  std::vector<Block<3>> blocks;
  blocks.reserve(num_blocks_);

  for (size_t layer = 0; layer < num_layers; ++layer) {
    const double z_lower = z_bounds[layer];
    const double z_upper = z_bounds[layer + 1];

    for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
      const size_t block_id = layer * num_radial_blocks + radial;
      const double inner_r =
          radial == 0 ? 0.0 : radial_partitioning_[radial - 1];
      const double outer_r = radial == num_radial_blocks - 1
                                 ? outer_radius_
                                 : radial_partitioning_[radial];

      // Map: (xi, eta, zeta) in [-1,1]^3
      //   xi -> r in [inner_r, outer_r]  (Affine)
      //   eta -> phi in [0, 2pi)         (Identity<1>, passes through)
      //   zeta -> z in [z_lower, z_upper] (Interval)
      // Then PolarToCartesian x Identity<1> maps (r, phi, z) -> (x, y, z)
      auto coord_map =
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Interval>{
                  Affine{-1.0, 1.0, inner_r, outer_r}, Identity1D{},
                  Interval{-1.0, 1.0, z_lower, z_upper,
                           distribution_in_z_[layer]}},
              CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                             Identity1D>{
                  CoordinateMaps::PolarToCartesian{}, Identity1D{}});

      DirectionMap<3, BlockNeighbors<3>> neighbors{};

      // Radial neighbors (xi direction)
      if (radial > 0) {
        neighbors.emplace(
            Direction<3>::lower_xi(),
            BlockNeighbors<3>(layer * num_radial_blocks + radial - 1, aligned));
      }
      if (radial < num_radial_blocks - 1) {
        neighbors.emplace(
            Direction<3>::upper_xi(),
            BlockNeighbors<3>(layer * num_radial_blocks + radial + 1, aligned));
      }

      // z neighbors (zeta direction)
      if (layer > 0) {
        neighbors.emplace(
            Direction<3>::lower_zeta(),
            BlockNeighbors<3>((layer - 1) * num_radial_blocks + radial,
                              aligned));
      } else if (is_periodic_in_z_) {
        // Connect bottom of layer 0 to top of last layer
        neighbors.emplace(
            Direction<3>::lower_zeta(),
            BlockNeighbors<3>((num_layers - 1) * num_radial_blocks + radial,
                              aligned));
      }
      if (layer < num_layers - 1) {
        neighbors.emplace(
            Direction<3>::upper_zeta(),
            BlockNeighbors<3>((layer + 1) * num_radial_blocks + radial,
                              aligned));
      } else if (is_periodic_in_z_) {
        // Connect top of last layer to bottom of layer 0
        neighbors.emplace(Direction<3>::upper_zeta(),
                          BlockNeighbors<3>(radial, aligned));
      }

      const auto& topology = radial == 0
                                 ? domain::topologies::full_cylinder
                                 : domain::topologies::cylindrical_shell;

      blocks.emplace_back(std::move(coord_map), block_id, std::move(neighbors),
                          block_names_.at(block_id), topology);
    }
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
AngularCylinder::external_boundary_conditions() const {
  if (mantle_boundary_condition_ == nullptr) {
    return {};
  }
  const size_t num_radial_blocks = 1 + radial_partitioning_.size();
  const size_t num_layers = 1 + partitioning_in_z_.size();

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};

  for (size_t layer = 0; layer < num_layers; ++layer) {
    for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
      const size_t block_id = layer * num_radial_blocks + radial;

      // Lower z boundary (lowermost layer only, non-periodic)
      if (not is_periodic_in_z_ and layer == 0 and
          lower_z_boundary_condition_ != nullptr) {
        boundary_conditions[block_id][Direction<3>::lower_zeta()] =
            lower_z_boundary_condition_->get_clone();
      }

      // Upper z boundary (uppermost layer only, non-periodic)
      if (not is_periodic_in_z_ and layer == num_layers - 1 and
          upper_z_boundary_condition_ != nullptr) {
        boundary_conditions[block_id][Direction<3>::upper_zeta()] =
            upper_z_boundary_condition_->get_clone();
      }

      // Radial (mantle) boundary on the outermost radial block
      if (radial == num_radial_blocks - 1) {
        boundary_conditions[block_id][Direction<3>::upper_xi()] =
            mantle_boundary_condition_->get_clone();
      }
    }
  }
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> AngularCylinder::initial_extents() const {
  // Throughout the code, we require n_phi be odd for numerical stability
  // We also require the angular modal space of the angular dimension to be <=
  // the angular modal space of the radial dimension. Here we set them to be
  // equal
  // \Phi angular modal max M = N_\Phi / 2 (integer division)
  // r angular modal max M = 2 * N_r - 2
  const size_t num_layers = 1 + partitioning_in_z_.size();

  // The center disk block: n_r derived from theta grid points (Zernike),
  // theta grid points, z grid points
  const size_t theta_M = initial_cylinder_theta_grid_points_ / 2;
  const size_t disk_n_r = theta_M / 2 + 1 + theta_M % 2;

  std::vector<std::array<size_t, 3>> extents;
  extents.reserve(num_blocks_);

  for (size_t layer = 0; layer < num_layers; ++layer) {
    // Center disk block
    extents.emplace_back(
        std::array<size_t, 3>{disk_n_r, initial_cylinder_theta_grid_points_,
                              initial_cylinder_z_grid_points_});
    // Shell blocks
    for (size_t shell = 0; shell < radial_partitioning_.size(); ++shell) {
      const auto& shell_pts = initial_hollow_cylinder_grid_points_[shell];
      extents.emplace_back(
          std::array<size_t, 3>{shell_pts[0], shell_pts[1], shell_pts[2]});
    }
  }
  return extents;
}

std::vector<std::array<size_t, 3>> AngularCylinder::initial_refinement_levels()
    const {
  // ZernikeB2 and Fourier should never be refined.
  const size_t num_radial_blocks = 1 + radial_partitioning_.size();
  const size_t num_layers = 1 + partitioning_in_z_.size();

  std::vector<std::array<size_t, 3>> refinement_levels;
  refinement_levels.reserve(num_blocks_);

  for (size_t layer = 0; layer < num_layers; ++layer) {
    for (size_t radial = 0; radial < num_radial_blocks; ++radial) {
      // Must use 0 for radial and theta refinement, but using the
      // specified z refinement for each layer
      refinement_levels.push_back({0, 0, initial_refinement_in_z_[layer]});
    }
  }
  return refinement_levels;
}
}  // namespace domain::creators
