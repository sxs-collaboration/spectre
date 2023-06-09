// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/Sphere.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/None.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame

namespace domain::creators {

namespace detail {
Excision::Excision(
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        local_boundary_condition)
    : boundary_condition(std::move(local_boundary_condition)) {}
}  // namespace detail

namespace {
struct DistributionVisitor {
  size_t num_shells;

  std::vector<domain::CoordinateMaps::Distribution> operator()(
      const domain::CoordinateMaps::Distribution distribution) const {
    return std::vector<domain::CoordinateMaps::Distribution>(num_shells,
                                                             distribution);
  }

  std::vector<domain::CoordinateMaps::Distribution> operator()(
      const std::vector<domain::CoordinateMaps::Distribution>& distributions)
      const {
    return distributions;
  }
};
}  // namespace

Sphere::Sphere(
    double inner_radius, double outer_radius,
    std::variant<Excision, InnerCube> interior,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_number_of_grid_points,
    bool use_equiangular_map,
    std::optional<EquatorialCompressionOptions> equatorial_compression,
    std::vector<double> radial_partitioning,
    const typename RadialDistribution::type& radial_distribution,
    ShellWedges which_wedges,
    std::optional<TimeDepOptionType> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : inner_radius_(inner_radius),
      outer_radius_(outer_radius),
      interior_(std::move(interior)),
      fill_interior_(std::holds_alternative<InnerCube>(interior_)),
      use_equiangular_map_(use_equiangular_map),
      equatorial_compression_(equatorial_compression),
      radial_partitioning_(std::move(radial_partitioning)),
      which_wedges_(which_wedges),
      time_dependent_options_(std::move(time_dependent_options)),
      outer_boundary_condition_(std::move(outer_boundary_condition)) {
  if (inner_radius_ > outer_radius_) {
    PARSE_ERROR(context,
                "Inner radius must be smaller than outer radius, but inner "
                "radius is " +
                    std::to_string(inner_radius_) + " and outer radius is " +
                    std::to_string(outer_radius_) + ".");
  }

  // Validate radial partitions
  if (not std::is_sorted(radial_partitioning_.begin(),
                         radial_partitioning_.end())) {
    PARSE_ERROR(context,
                "Specify radial partitioning in ascending order. Specified "
                "radial partitioning is: " +
                    get_output(radial_partitioning_));
  }
  if (not radial_partitioning_.empty()) {
    if (radial_partitioning_.front() <= inner_radius_) {
      PARSE_ERROR(context,
                  "First radial partition must be larger than inner "
                  "radius, but is: " +
                      std::to_string(inner_radius_));
    }
    if (radial_partitioning_.back() >= outer_radius_) {
      PARSE_ERROR(context,
                  "Last radial partition must be smaller than outer "
                  "radius, but is: " +
                      std::to_string(outer_radius_));
    }
    const auto duplicate = std::adjacent_find(radial_partitioning_.begin(),
                                              radial_partitioning_.end());
    if (duplicate != radial_partitioning_.end()) {
      PARSE_ERROR(context, "Radial partitioning contains duplicate element: "
                               << *duplicate);
    }
  }
  num_shells_ = 1 + radial_partitioning_.size();
  radial_distribution_ =
      std::visit(DistributionVisitor{num_shells_}, radial_distribution);
  if (radial_distribution_.size() != num_shells_) {
    PARSE_ERROR(context,
                "Specify a 'RadialDistribution' for every spherical shell. You "
                "specified "
                    << radial_distribution_.size()
                    << " items, but the domain has " << num_shells_
                    << " shells.");
  }
  if (fill_interior_ and radial_distribution_.front() !=
                             domain::CoordinateMaps::Distribution::Linear) {
    PARSE_ERROR(context,
                "The 'RadialDistribution' must be 'Linear' for the innermost "
                "shell filled with a cube because it changes in sphericity. "
                "Add entries to 'RadialPartitioning' to add outer shells for "
                "which you can select different radial distributions.");
  }

  // Determine number of blocks
  num_blocks_per_shell_ =
      which_wedges_ == ShellWedges::All
          ? 6
          : which_wedges_ == ShellWedges::FourOnEquator ? 4 : 1;
  num_blocks_ = num_blocks_per_shell_ * num_shells_ + (fill_interior_ ? 1 : 0);

  // Create block names and groups
  static std::array<std::string, 6> wedge_directions{
      "UpperZ", "LowerZ", "UpperY", "LowerY", "UpperX", "LowerX"};
  for (size_t shell = 0; shell < num_shells_; ++shell) {
    std::string shell_prefix = "Shell" + std::to_string(shell);
    for (size_t direction = which_wedge_index(which_wedges_); direction < 6;
         ++direction) {
      const std::string wedge_name =
          shell_prefix + gsl::at(wedge_directions, direction);
      block_names_.emplace_back(wedge_name);
      if (num_shells_ > 1) {
        block_groups_[shell_prefix].insert(wedge_name);
      }
      block_groups_["Wedges"].insert(wedge_name);
    }
  }
  if (fill_interior_) {
    block_names_.emplace_back("InnerCube");
  }
  ASSERT(block_names_.size() == num_blocks_,
         "Invalid number of block names. Should be "
             << num_blocks_ << " but is " << block_names_.size() << ".");

  // Expand initial refinement and number of grid points over all blocks
  const ExpandOverBlocks<size_t, 3> expand_over_blocks{block_names_,
                                                       block_groups_};
  try {
    initial_refinement_ = std::visit(expand_over_blocks, initial_refinement);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  try {
    initial_number_of_grid_points_ =
        std::visit(expand_over_blocks, initial_number_of_grid_points);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  if (fill_interior_) {
    // The central cube has no notion of a "radial" direction, so we set
    // refinement and number of grid points of the central cube z direction to
    // its y value, which corresponds to the azimuthal direction of the
    // wedges. This keeps the boundaries conforming when the radial direction
    // is chosen differently to the angular directions.
    auto& central_cube_refinement = initial_refinement_.back();
    auto& central_cube_grid_points = initial_number_of_grid_points_.back();
    central_cube_refinement[2] = central_cube_refinement[1];
    central_cube_grid_points[2] = central_cube_grid_points[1];
  }

  // Validate boundary conditions
  using domain::BoundaryConditions::is_none;
  if (is_none(outer_boundary_condition_) or
      (not fill_interior_ and
       is_none(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(
        context,
        "None boundary condition is not supported. If you would like an "
        "outflow-type boundary condition, you must use that.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_) or
      (not fill_interior_ and
       is_periodic(std::get<Excision>(interior_).boundary_condition))) {
    PARSE_ERROR(context,
                "Cannot have periodic boundary conditions with a Sphere");
  }
  // Validate consistency of inner and outer boundary condition
  if (not fill_interior_) {
    const auto& inner_boundary_condition =
        std::get<Excision>(interior_).boundary_condition;
    if ((inner_boundary_condition == nullptr) !=
        (outer_boundary_condition_ == nullptr)) {
      PARSE_ERROR(
          context,
          "Must specify either both inner and outer boundary conditions "
          "or neither.");
    }
  }
  if (outer_boundary_condition != nullptr and
      which_wedges_ != ShellWedges::All) {
    PARSE_ERROR(
        context,
        "Can only apply boundary conditions when the outer boundary of the "
        "domain is a full sphere. "
        "Additional cases can be supported by adding them to the Sphere "
        "domain creator.");
  }

  if (time_dependent_options_.has_value()) {
    use_hard_coded_maps_ =
        std::holds_alternative<sphere::TimeDependentMapOptions>(
            time_dependent_options_.value());

    if (use_hard_coded_maps_) {
      if (fill_interior_) {
        PARSE_ERROR(context,
                    "Currently cannot use hard-coded time dependent maps with "
                    "an inner cube. Use a TimeDependence instead.");
      }

      // Build the maps. We only apply the maps in the inner-most shell. The
      // inner radius is what's passed in, but the outer radius is the outer
      // radius of the inner-most shell so we have to see how many shells we
      // have
      std::get<sphere::TimeDependentMapOptions>(time_dependent_options_.value())
          .build_maps(std::array{0.0, 0.0, 0.0}, inner_radius_,
                      radial_partitioning_.empty() ? outer_radius_
                                                   : radial_partitioning_[0]);
    }
  }
}

Domain<3> Sphere::create_domain() const {
  std::vector<std::array<size_t, 8>> corners =
      corners_for_radially_layered_domains(num_shells_, fill_interior_,
                                           {{1, 2, 3, 4, 5, 6, 7, 8}},
                                           which_wedges_);

  auto coord_maps = domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                            Frame::Inertial, 3>(
      sph_wedge_coordinate_maps(
          inner_radius_, outer_radius_,
          fill_interior_ ? std::get<InnerCube>(interior_).sphericity : 1.0, 1.0,
          use_equiangular_map_, false, radial_partitioning_,
          radial_distribution_, which_wedges_));

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  if (fill_interior_) {
    const double inner_cube_sphericity =
        std::get<InnerCube>(interior_).sphericity;
    if (inner_cube_sphericity == 0.0) {
      if (use_equiangular_map_) {
        coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Equiangular3D{
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                    Equiangular(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))}));
      } else {
        coord_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0)),
                         Affine(-1.0, 1.0, -1.0 * inner_radius_ / sqrt(3.0),
                                inner_radius_ / sqrt(3.0))}));
      }
    } else {
      coord_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              BulgedCube{inner_radius_, inner_cube_sphericity,
                         use_equiangular_map_}));
    }
  } else {
    // Set up excision sphere only for ShellWedges::All
    // - The first 6 blocks enclose the excised sphere, see
    //   sph_wedge_coordinate_maps
    // - The 3D wedge map is oriented such that the lower-zeta logical direction
    //   points radially inward.
    if (which_wedges_ == ShellWedges::All) {
      excision_spheres.emplace(
          "ExcisionSphere",
          ExcisionSphere<3>{inner_radius_,
                            tnsr::I<double, 3, Frame::Grid>{0.0},
                            {{0, Direction<3>::lower_zeta()},
                             {1, Direction<3>::lower_zeta()},
                             {2, Direction<3>::lower_zeta()},
                             {3, Direction<3>::lower_zeta()},
                             {4, Direction<3>::lower_zeta()},
                             {5, Direction<3>::lower_zeta()}}});
    }
  }

  Domain<3> domain{std::move(coord_maps),       corners,      {},
                   std::move(excision_spheres), block_names_, block_groups_};
  ASSERT(domain.blocks().size() == num_blocks_,
         "Unexpected number of blocks. Expected "
             << num_blocks_ << " but created " << domain.blocks().size()
             << ".");

  if (time_dependent_options_.has_value()) {
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        block_maps_grid_to_inertial{num_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        block_maps_grid_to_distorted{num_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        block_maps_distorted_to_inertial{num_blocks_};

    if (use_hard_coded_maps_) {
      const auto& hard_coded_options =
          std::get<sphere::TimeDependentMapOptions>(
              time_dependent_options_.value());

      // First shell gets the distorted maps.
      for (size_t block_id = 0; block_id < num_blocks_; block_id++) {
        const bool include_distorted_map_in_first_shell =
            block_id < num_blocks_per_shell_;
        block_maps_grid_to_distorted[block_id] =
            hard_coded_options.grid_to_distorted_map(
                include_distorted_map_in_first_shell);
        block_maps_distorted_to_inertial[block_id] =
            hard_coded_options.distorted_to_inertial_map(
                include_distorted_map_in_first_shell);
        block_maps_grid_to_inertial[block_id] =
            hard_coded_options.grid_to_inertial_map(
                include_distorted_map_in_first_shell);
      }
    } else {
      const auto& time_dependence = std::get<std::unique_ptr<
          domain::creators::time_dependence::TimeDependence<3>>>(
          time_dependent_options_.value());

      block_maps_grid_to_inertial =
          time_dependence->block_maps_grid_to_inertial(num_blocks_);
      block_maps_grid_to_distorted =
          time_dependence->block_maps_grid_to_distorted(num_blocks_);
      block_maps_distorted_to_inertial =
          time_dependence->block_maps_distorted_to_inertial(num_blocks_);
    }

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
Sphere::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }

  // This assumes 6 wedges making up the shell. If you need to support other
  // configurations the below code needs to be updated. This would require
  // adding more boundary condition options to the domain creator.
  if (which_wedges_ != ShellWedges::All) {
    ERROR(
        "Boundary conditions for incomplete spherical shells are not currently "
        "implemented. Add support to the Sphere domain creator.");
  }

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{num_blocks_};
  // Outer boundary conditions
  const size_t outer_blocks_offset = num_blocks_ - 6 - (fill_interior_ ? 1 : 0);
  for (size_t i = 0; i < 6; ++i) {
    boundary_conditions[i + outer_blocks_offset][Direction<3>::upper_zeta()] =
        outer_boundary_condition_->get_clone();
  }
  // Inner boundary conditions
  if (not fill_interior_) {
    const auto& inner_boundary_condition =
        std::get<Excision>(interior_).boundary_condition;
    for (size_t i = 0; i < 6; ++i) {
      boundary_conditions[i][Direction<3>::lower_zeta()] =
          inner_boundary_condition->get_clone();
    }
  }
  return boundary_conditions;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
Sphere::functions_of_time(const std::unordered_map<std::string, double>&
                              initial_expiration_times) const {
  if (time_dependent_options_.has_value()) {
    if (use_hard_coded_maps_) {
      return std::get<sphere::TimeDependentMapOptions>(
                 time_dependent_options_.value())
          .create_functions_of_time(inner_radius_, initial_expiration_times);
    } else {
      return std::get<std::unique_ptr<
          domain::creators::time_dependence::TimeDependence<3>>>(
                 time_dependent_options_.value())
          ->functions_of_time(initial_expiration_times);
    }
  } else {
    return {};
  }
}
}  // namespace domain::creators
