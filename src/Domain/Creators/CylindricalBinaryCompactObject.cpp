// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/CylindricalBinaryCompactObject.hpp"

#include <cmath>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalFlatEndcap.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ExpandOverBlocks.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "NumericalAlgorithms/RootFinding/QuadraticEquation.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/MakeArray.hpp"

namespace {
std::array<double, 3> rotate_to_z_axis(const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::lower_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::upper_xi()}},
      input);
}
std::array<double, 3> rotate_from_z_to_x_axis(
    const std::array<double, 3> input) {
  return discrete_rotation(
      OrientationMap<3>{std::array<Direction<3>, 3>{Direction<3>::upper_zeta(),
                                                    Direction<3>::upper_eta(),
                                                    Direction<3>::lower_xi()}},
      input);
}
std::array<double, 3> flip_about_xy_plane(const std::array<double, 3> input) {
  return std::array<double, 3>{input[0], input[1], -input[2]};
}
}  // namespace

namespace domain::creators {
CylindricalBinaryCompactObject::CylindricalBinaryCompactObject(
    std::array<double, 3> center_A, std::array<double, 3> center_B,
    double radius_A, double radius_B, bool include_inner_sphere_A,
    bool include_inner_sphere_B, double outer_radius,
    const typename InitialRefinement::type& initial_refinement,
    const typename InitialGridPoints::type& initial_grid_points,
    std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        inner_boundary_condition,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : center_A_(rotate_to_z_axis(center_A)),
      center_B_(rotate_to_z_axis(center_B)),
      radius_A_(radius_A),
      radius_B_(radius_B),
      include_inner_sphere_A_(include_inner_sphere_A),
      include_inner_sphere_B_(include_inner_sphere_B),
      outer_radius_(outer_radius),
      inner_boundary_condition_(std::move(inner_boundary_condition)),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependent_options_(std::move(time_dependent_options)) {
  if (center_A_[2] <= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterA is expected to be positive");
  }
  if (center_B_[2] >= 0.0) {
    PARSE_ERROR(
        context,
        "The x-coordinate of the input CenterB is expected to be negative");
  }
  if (radius_A_ <= 0.0 or radius_B_ <= 0.0) {
    PARSE_ERROR(context, "RadiusA and RadiusB are expected to be positive");
  }
  if (radius_A_ < radius_B_) {
    PARSE_ERROR(context, "RadiusA should not be smaller than RadiusB");
  }
  if (std::abs(center_A_[2]) > std::abs(center_B_[2])) {
    PARSE_ERROR(context,
                "We expect |x_A| <= |x_B|, for x the x-coordinate of either "
                "CenterA or CenterB.  We should roughly have "
                "RadiusA x_A + RadiusB x_B = 0 (i.e. for BBHs the "
                "center of mass should be about at the origin).");
  }
  // The value 3.0 * (center_A_[2] - center_B_[2]) is what is
  // chosen in SpEC as the inner radius of the innermost outer sphere.
  if (outer_radius_ < 3.0 * (center_A_[2] - center_B_[2])) {
    PARSE_ERROR(context,
                "OuterRadius is too small. Please increase it "
                "beyond "
                    << 3.0 * (center_A_[2] - center_B_[2]));
  }

  if ((outer_boundary_condition_ == nullptr) xor
      (inner_boundary_condition_ == nullptr)) {
    PARSE_ERROR(context,
                "Must specify either both inner and outer boundary conditions "
                "or neither.");
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(inner_boundary_condition_) or
      is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  // The choices made below for the quantities xi, z_cutting_plane_,
  // and xi_min_sphere_e are the ones made in SpEC, and in the
  // Appendix of https://arxiv.org/abs/1206.3015.  Other choices could
  // be made that would still result in a reasonable Domain. In
  // particular, during a SpEC BBH evolution the excision boundaries
  // can sometimes get too close to z_cutting_plane_, and the
  // simulation must be halted and regridded with a different choice
  // of z_cutting_plane_, so it may be possible to choose a different
  // initial value of z_cutting_plane_ that reduces the number of such
  // regrids or eliminates them.

  // xi is the quantity in Eq. (A10) of
  // https://arxiv.org/abs/1206.3015 that represents how close the
  // cutting plane is to either center.  Unfortunately, there is a
  // discrepancy between what xi means in the paper and what it is in
  // the code.  I (Mark) think that this is a typo in the paper,
  // because otherwise the domain doesn't make sense.  To fix this,
  // either Eq. (A9) in the paper should have xi -> 1-xi, or Eq. (A10)
  // should have x_A and x_B swapped.
  // Here we will use the same definition of xi in Eq. (A10), but we
  // will swap xi -> 1-xi in Eq. (A9).
  // Therefore, xi = 0 means that the cutting plane passes through the center of
  // object B, and xi = 1 means that the cutting plane passes through
  // the center of object A.  Note that for |x_A| <= |x_B| (as assumed
  // above), xi is always <= 1/2.
  constexpr double xi_min = 0.25;
  // Same as Eq. (A10)
  const double xi =
      std::max(xi_min, std::abs(center_A_[2]) /
                           (std::abs(center_A_[2]) + std::abs(center_B_[2])));

  // Compute cutting plane
  // This is Eq. (A9) with xi -> 1-xi.
  z_cutting_plane_ = cut_spheres_offset_factor_ *
                     ((1.0 - xi) * center_B_[2] + xi * center_A_[2]);

  // outer_radius_A is the outer radius of the inner sphere A, if it exists.
  // If the inner sphere A does not exist, then outer_radius_A is the same
  // as radius_A_.
  // If the inner sphere does exist, the algorithm for computing
  // outer_radius_A is the same as in SpEC when there is one inner shell.
  outer_radius_A_ =
      include_inner_sphere_A_
          ? radius_A_ +
                0.5 * (std::abs(z_cutting_plane_ - center_A_[2]) - radius_A_)
          : radius_A_;

  // outer_radius_B is the outer radius of the inner sphere B, if it exists.
  // If the inner sphere B does not exist, then outer_radius_B is the same
  // as radius_B_.
  // If the inner sphere does exist, the algorithm for computing
  // outer_radius_B is the same as in SpEC when there is one inner shell.
  outer_radius_B_ =
      include_inner_sphere_B_
          ? radius_B_ +
                0.5 * (std::abs(z_cutting_plane_ - center_B_[2]) - radius_B_)
          : radius_B_;

  // Add SphereE blocks if necessary.  Note that
  // https://arxiv.org/abs/1206.3015 has a mistake just above
  // Eq. (A.11) and the same mistake above Eq. (A.20), where it lists
  // the wrong mass ratio (for BBHs). The correct statement is that if
  // xi <= 1/3, this means that the mass ratio (for BBH) is large (>=2)
  // and we should add SphereE blocks.
  constexpr double xi_min_sphere_e = 1.0 / 3.0;
  if (xi <= xi_min_sphere_e) {
    // The following ERROR will be removed in an upcoming PR that
    // will support higher mass ratios.
    ERROR(
        "We currently only support domains where objects A and B are "
        "approximately the same size, and approximately the same distance from "
        "the origin.  More technically, we support xi > "
        << xi_min_sphere_e << ", but the value of xi is " << xi
        << ". Support for more general domains will be added in the near "
           "future");
  }

  // Create grid anchors in x direction from unrotated input centers
  grid_anchors_ = bco::create_grid_anchors(center_A, center_B);

  // Build the set of cylindrical block groups and block names so the
  // validation below can distinguish spherical-harmonic blocks from other
  // blocks and block groups.
  std::unordered_set<std::string> filled_cylinder_names{};
  std::unordered_set<std::string> hollow_cylinder_names{};

  // Create cylinder block names and groups
  auto add_filled_cylinder_name = [this, &filled_cylinder_names](
                                      const std::string& prefix,
                                      const std::string& group_name) {
    const std::string name = std::string(prefix).append("FilledCylinder");
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    filled_cylinder_names.insert(name);
    filled_cylinder_names.insert(group_name);
  };
  auto add_cylinder_name = [this, &hollow_cylinder_names](
                               const std::string& prefix,
                               const std::string& group_name) {
    const std::string name = std::string(prefix).append("Cylinder");
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    hollow_cylinder_names.insert(name);
    hollow_cylinder_names.insert(group_name);
  };

  // CA Filled Cylinder
  add_filled_cylinder_name("CA", "Outer");

  // CA Cylinder
  add_cylinder_name("CA", "Outer");

  // EA Filled Cylinder
  add_filled_cylinder_name("EA", "InnerA");

  // EA Cylinder
  add_cylinder_name("EA", "InnerA");

  // EB Filled Cylinder
  add_filled_cylinder_name("EB", "InnerB");

  // EB Cylinder
  add_cylinder_name("EB", "InnerB");

  // MA Filled Cylinder
  add_filled_cylinder_name("MA", "InnerA");

  // MB Filled Cylinder
  add_filled_cylinder_name("MB", "InnerB");

  // CB Filled Cylinder
  add_filled_cylinder_name("CB", "Outer");

  // CB Cylinder
  add_cylinder_name("CB", "Outer");

  // combine filled and hollow cylinder blocks and groups into one set
  std::unordered_set<std::string> all_cylinder_names;
  all_cylinder_names.insert(std::begin(filled_cylinder_names),
                            std::end(filled_cylinder_names));
  all_cylinder_names.insert(std::begin(hollow_cylinder_names),
                            std::end(hollow_cylinder_names));

  // Build the set of spherical-harmonic shell block groups and block names so
  // the validation below can distinguish spherical-harmonic blocks from other
  // blocks and block groups.
  std::unordered_set<std::string> spherical_harmonic_shell_names{};

  // Create block names and groups
  auto add_spherical_shell_name = [this, &spherical_harmonic_shell_names](
                                      const std::string& prefix,
                                      const std::string& group_name,
                                      const size_t shell_number) {
    const std::string name = std::string(prefix).append("Shell").append(
        std::to_string(shell_number));
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
    block_positions_[name] = block_names_.size() - 1;
    spherical_harmonic_shell_names.insert(name);
    spherical_harmonic_shell_names.insert(group_name);
  };

  if (include_inner_sphere_A) {
    add_spherical_shell_name("InnerA", "InnerSphereA", 0);
  }
  if (include_inner_sphere_B) {
    add_spherical_shell_name("InnerB", "InnerSphereB", 0);
  }
  add_spherical_shell_name("Outer", "OuterSphere", 0);

  number_of_blocks_ = block_names_.size();
  ASSERT(number_of_blocks_ == block_positions_.size(),
         "Size of block_positions_ map should be equal to the number of blocks "
         "in the domain.");

  // Since BinaryCompactObject::InitialGridPoints type differs from
  // CylindricalBinaryCompactObject::InitialGridPoints type, need to first
  // create the BCO-compatible type with the CBCO data to be able to reuse the
  // functionality of bco::validate_initial_grid_points() and
  // bco::set_initial_grid_points().
  const auto bco_initial_grid_points = std::visit(
      [](const auto& value) {
        return BinaryCompactObject::InitialGridPoints::type{value};
      },
      initial_grid_points);
  // Validate that the input file has the correct format for
  // InitialGridPoints. No need to validate the format for InitialRefinement
  // because it does not accept a map of strings to possibly
  // differently-sized arrays for refinement. If a map is provided, it already
  // only accepts a map of size_t keys.
  bco::validate_initial_grid_points(context, bco_initial_grid_points,
                                    spherical_harmonic_shell_names,
                                    filled_cylinder_names);

  // For expanding initial refinement and grid points over all blocks
  const ExpandOverBlocks<std::array<size_t, 3>> expand_over_blocks{
      block_names_, block_groups_};
  try {
    // Since BinaryCompactObject::InitialRefinement map type differs from
    // CylindricalBinaryCompactObject::InitialRefinement map type, need to first
    // create the BCO-compatible type with the CBCO data to be able to reuse the
    // functionality of bco::set_initial_refinement().
    using bco_ref_map_type =
        std::unordered_map<std::string,
                           std::variant<std::array<size_t, 3>, size_t>>;
    using cbco_ref_map_type = std::unordered_map<std::string, size_t>;
    const auto bco_initial_refinement =
        std::holds_alternative<size_t>(initial_refinement)
            ? BinaryCompactObject::InitialRefinement::type{std::get<size_t>(
                  initial_refinement)}
            : BinaryCompactObject::InitialRefinement::type{bco_ref_map_type{
                  std::get<cbco_ref_map_type>(initial_refinement).begin(),
                  std::get<cbco_ref_map_type>(initial_refinement).end()}};
    initial_refinement_ = bco::set_initial_refinement(
        expand_over_blocks, bco_initial_refinement,
        spherical_harmonic_shell_names, all_cylinder_names);
    // If a global single-number h-refinement was used, post-process the
    // expanded cylinder and spherical shell blocks to make the angular
    // directions have h refinement = 0.
    if (std::holds_alternative<size_t>(initial_refinement)) {
      for (const auto& [name, position] : block_positions_) {
        if (name.find("Cylinder") != std::string::npos) {
          // Set cylinder h refinement to {0, 0, z}
          initial_refinement_[position][0] = 0;
          initial_refinement_[position][1] = 0;
        } else if (name.find("Shell") != std::string::npos) {
          // Set spherical shell h refinement to {r, 0, 0}
          initial_refinement_[position][1] = 0;
          initial_refinement_[position][2] = 0;
        }
      }
    }
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }

  // Validate angular h-refinement == 0 in cylinder and spherical shell blocks
  for (const auto& [name, position] : block_positions_) {
    if (name.find("Cylinder") != std::string::npos) {
      if (gsl::at(gsl::at(initial_refinement_, position), 0) != 0 or
          gsl::at(gsl::at(initial_refinement_, position), 1) != 0) {
        PARSE_ERROR(context,
                    "Angular h-refinement is not supported for cylindrical "
                    "blocks. Specify refinement for "
                        << name << " as a single number.");
      }
    } else if (name.find("Shell") != std::string::npos) {
      if (gsl::at(gsl::at(initial_refinement_, position), 1) != 0 or
          gsl::at(gsl::at(initial_refinement_, position), 2) != 0) {
        PARSE_ERROR(context,
                    "Angular h-refinement is not supported for "
                    "spherical-harmonic shell blocks. Specify refinement for "
                        << name << " as a single number.");
      }
    }
  }

  try {
    initial_grid_points_ = bco::set_initial_grid_points(
        expand_over_blocks, bco_initial_grid_points,
        spherical_harmonic_shell_names, filled_cylinder_names);
    // If a global single-number p-refinement was used, post-process the
    // expanded filled cylinder blocks to make the angular directions have the
    // correct number of spectral points for ZernikeB2.
    if (std::holds_alternative<size_t>(initial_grid_points)) {
      for (const auto& [name, position] : block_positions_) {
        if (name.find("FilledCylinder") != std::string::npos) {
          // note for ZernikeB2:
          // radial_points = (theta_modes / 2) + 1 + (theta_modes % 2), so one
          // could have odd or even theta_modes for the same radial_points.
          // Choosing the even theta_modes for the same radial_points means:
          //   theta_modes = 2 * (radial_points - 1)
          //   theta_points = 2 * theta_modes + 1 = 4 * radial_modes - 3
          // Choosing the odd theta_modes for the same radial_points means:
          //   theta_modes = 2 * (radial_points - 2) + 1
          //   theta_points = 2 * theta_modes + 1 = 4 * radial_modes - 5
          // Here, we choose the even case to get one extra theta_mode out of
          // radial_points.
          initial_grid_points_[position][1] =
              4 * gsl::at(gsl::at(initial_grid_points_, position), 0) - 3;
        }
      }
    }
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }

  // Validate p-refinement values in cylinder and spherical shell blocks
  for (const auto& [name, position] : block_positions_) {
    if (name.find("FilledCylinder") != std::string::npos) {
      // Validate number of radial points for filled cylinders is > 2
      if (gsl::at(gsl::at(initial_grid_points_, position), 0) <= 2) {
        PARSE_ERROR(context,
                    "Filled cylindrical block "
                        << name
                        << " must have more than 2 radial grid points.");
      }

      // Validate number of angular grid points in filled cylinder blocks is
      // what is expected by ZernikeB2. The Zernike disk is fully specified by
      // either the number of radial points or the number of theta points, so
      // check that they relate as expected.
      const size_t num_theta_modes =
          gsl::at(gsl::at(initial_grid_points_, position), 1) / 2;
      const size_t expected_num_r_points =
          (num_theta_modes / 2) + 1 + (num_theta_modes % 2);
      if (gsl::at(gsl::at(initial_grid_points_, position), 0) !=
          expected_num_r_points) {
        PARSE_ERROR(context,
                    "Filled cylinder blocks must have "
                    "num_r_points = ((num_theta_points / 2) / 2) + 1 + "
                    "((num_theta_points / 2) % 2). Specify grid points for "
                        << name << " as [num_radial_points, num_z_points].");
      }
    }
    if (name.find("Cylinder") != std::string::npos) {
      // Validate number of angular grid points in all cylindrical blocks are
      // odd.
      if (gsl::at(gsl::at(initial_grid_points_, position), 1) % 2 == 0) {
        PARSE_ERROR(context,
                    "Cylindrical block "
                        << name
                        << " must have an odd number of angular grid points.");
      }
    } else if (name.find("Shell") != std::string::npos) {
      // For spherical-harmonic shell blocks, initial_number_of_grid_points_
      // stores {n_radial, l_max, m_max}. First validate that l_max == m_max,
      // then convert (l_max, m_max) to the number of collocation points the
      // spherical-harmonic basis uses in each angular direction.
      const size_t l_max = gsl::at(gsl::at(initial_grid_points_, position), 1);
      const size_t m_max = gsl::at(gsl::at(initial_grid_points_, position), 2);
      if (l_max != m_max) {
        PARSE_ERROR(context,
                    "Spherical-harmonic shell blocks must have L_max = M_max. "
                    "Specify grid points for "
                        << name << " as [radial_points, L_max].");
      }
      initial_grid_points_[position][1] = ylm::Spherepack::n_theta_points(
          gsl::at(gsl::at(initial_grid_points_, position), 1));
      initial_grid_points_[position][2] = ylm::Spherepack::n_phi_points(
          gsl::at(gsl::at(initial_grid_points_, position), 2));
    }
  }

  // Build time-dependent maps
  // The size map, which is applied from the grid to distorted frame, currently
  // needs to start and stop at certain radii around each excision. If the inner
  // spheres aren't included, the outer radii would have to be in the middle of
  // a block. With the inner spheres, the outer radii can be at block
  // boundaries.
  if (time_dependent_options_.has_value() and
      not(include_inner_sphere_A and include_inner_sphere_B)) {
    PARSE_ERROR(context,
                "To use the CylindricalBBH domain with time-dependent maps, "
                "you must include the inner spheres for both objects. "
                "Currently, one or both objects is missing the inner sphere.");
  }

  if (time_dependent_options_.has_value()) {
    const double inner_common_radius = 3.0 * (center_A_[2] - center_B_[2]);
    const auto center_A_aligned = rotate_from_z_to_x_axis(center_A_);
    const auto center_B_aligned = rotate_from_z_to_x_axis(center_B_);
    time_dependent_options_->build_maps(
        std::array{center_A_aligned, center_B_aligned}, std::nullopt,
        std::nullopt,
        std::array{z_cutting_plane_,
                   0.5 * (center_A_aligned[1] + center_B_aligned[1]),
                   0.5 * (center_A_aligned[2] + center_B_aligned[2])},
        std::array{radius_A_, outer_radius_A_},
        std::array{radius_B_, outer_radius_B_}, false, false,
        inner_common_radius, outer_radius_);
  }
}

Domain<3> CylindricalBinaryCompactObject::create_domain() const {
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};

  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const OrientationMap<3> rotate_to_minus_z_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::upper_eta(),
      Direction<3>::lower_zeta()}};

  const OrientationMap<3> aligned = OrientationMap<3>::create_aligned();

  const std::array<double, 3> center_cutting_plane = {0.0, 0.0,
                                                      z_cutting_plane_};

  // The labels EA, EB, EE, etc are from Figure 20 of
  // https://arxiv.org/abs/1206.3015
  //
  // center_EA and radius_EA are the center and outer-radius of the
  // cylindered-sphere EA in Figure 20.
  //
  // center_EB and radius_EB are the center and outer-radius of the
  // cylindered-sphere EB in Figure 20.
  //
  // radius_MB is eq. A16 or A23 in the paper (depending on whether
  // the EE spheres exist), and is the radius of the circle where the EB
  // sphere intersects the cutting plane.
  const std::array<double, 3> center_EA = {
      0.0, 0.0, cut_spheres_offset_factor_ * center_A_[2]};
  const std::array<double, 3> center_EB = {
      0.0, 0.0, center_B_[2] * cut_spheres_offset_factor_};
  const double radius_MB =
      std::abs(cut_spheres_offset_factor_ * center_B_[2] - z_cutting_plane_);
  const double radius_EA =
      sqrt(square(center_EA[2] - z_cutting_plane_) + square(radius_MB));
  const double radius_EB =
      sqrt(2.0) * std::abs(center_EB[2] - z_cutting_plane_);

  // Construct a coordinate map that goes from logical coordinates to a unit
  // right cylinder block. The radii and bounds are what are expected by the
  // UniformCylindricalEndCap and UniformCylindricalFlatEndCap maps.
  const double cylinder_inner_radius = 0.0;
  const double cylinder_outer_radius = 1.0;
  const double cylinder_lower_bound_z = -1.0;
  const double cylinder_upper_bound_z = 1.0;

  const auto logical_to_cylinder_map =
      cyl_coordinate_map(cylinder_inner_radius, cylinder_outer_radius,
                         cylinder_lower_bound_z, cylinder_upper_bound_z);

  // Lambda that takes a pre-rotation map, a UniformCylindricalEndcap or a
  // UniformCylindricalFlatEndcap map and a DiscreteRotation map, composes it
  // with the logical-to-cylinder map, and adds it to the list of
  // coordinate maps. Also adds boundary conditions if requested. The
  // pre-rotation map is used by blocks at the cutting plane to achieve nodal
  // alignment with their block neighbor on the other side of the plane.
  auto add_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_map](
          const CoordinateMaps::DiscreteRotation<3>& pre_rotation_map,
          const auto& endcap_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylinder_map = ::domain::push_back(
            ::domain::push_back(
                ::domain::push_back(logical_to_cylinder_map, pre_rotation_map),
                endcap_map),
            rotation_map);

        coordinate_maps.emplace_back(
            std::make_unique<
                std::decay_t<decltype(new_logical_to_cylinder_map)>>(
                std::move(new_logical_to_cylinder_map)));
      };

  // Construct a coordinate map that goes from logical coordinates to a unit
  // right cylindrical shell block. The radii and bounds are what are expected
  // by the UniformCylindricalSide map.
  const double cylindrical_shell_inner_radius = 1.0;
  const double cylindrical_shell_outer_radius = 2.0;
  const double cylindrical_shell_lower_bound_z = -1.0;
  const double cylindrical_shell_upper_bound_z = 1.0;

  const auto logical_to_cylindrical_shell_map = cyl_coordinate_map(
      cylindrical_shell_inner_radius, cylindrical_shell_outer_radius,
      cylindrical_shell_lower_bound_z, cylindrical_shell_upper_bound_z);

  // Lambda that takes a pre-rotation map, a UniformCylindricalSide map, and a
  // DiscreteRotation map, composes it with the logical-to-cylinder maps, and
  // adds it to the list of coordinate maps.  Also adds boundary conditions if
  // requested.  The pre-rotation map is used by blocks at the cutting plane to
  // achieve nodal alignment with their block neighbor on the other side of the
  // plane.
  auto add_side_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylindrical_shell_map](
          const CoordinateMaps::DiscreteRotation<3>& pre_rotation_map,
          const CoordinateMaps::UniformCylindricalSide& side_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylindrical_shell_map = ::domain::push_back(
            ::domain::push_back(
                ::domain::push_back(logical_to_cylindrical_shell_map,
                                    pre_rotation_map),
                side_map),
            rotation_map);

        coordinate_maps.emplace_back(
            std::make_unique<
                std::decay_t<decltype(new_logical_to_cylindrical_shell_map)>>(
                std::move(new_logical_to_cylindrical_shell_map)));
      };

  // Inner radius of the outer C shell.
  const double inner_radius_C = 3.0 * (center_A_[2] - center_B_[2]);

  // z_cut_CA_lower is the lower z_plane position for the CA endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_A.
  const double z_cut_CA_lower =
      z_cutting_plane_ + 1.6 * (center_EA[2] - z_cutting_plane_);
  // z_cut_CA_upper is the upper z_plane position for the CA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_CA_upper =
      std::max(0.5 * (z_cut_CA_lower + inner_radius_C), 0.7 * inner_radius_C);
  // z_cut_EA_upper is the upper z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_upper = center_A_[2] + 0.7 * outer_radius_A_;
  // z_cut_EA_lower is the lower z_plane position for the EA endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.
  const double z_cut_EA_lower = center_A_[2] - 0.7 * outer_radius_A_;

  // CA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalEndcap(center_EA, make_array<3>(0.0),
                                               radius_EA, inner_radius_C,
                                               z_cut_CA_lower, z_cut_CA_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CA Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalSide(
          // codecov complains about the next line being untested.
          // No idea why, since this entire function is called.
          // LCOV_EXCL_START
          center_EA, make_array<3>(0.0), radius_EA, inner_radius_C,
          // LCOV_EXCL_STOP
          z_cut_CA_lower, z_cutting_plane_, z_cut_CA_upper, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      CoordinateMaps::UniformCylindricalEndcap(center_A_, center_EA,
                                               outer_radius_A_, radius_EA,
                                               z_cut_EA_upper, z_cut_CA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalSide(  // LCOV_EXCL_LINE
          center_A_, center_EA, outer_radius_A_, radius_EA, z_cut_EA_upper,
          z_cut_EA_lower, z_cut_CA_lower, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // z_cut_CB_lower is the lower z_plane position for the CB endcap,
  // defined by https://arxiv.org/abs/1206.3015 in the bulleted list
  // after Eq. (A.19) EXCEPT that here we use a factor of 1.6 instead of 1.5
  // to put the plane farther from center_B.
  // Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_lower =
      z_cutting_plane_ + 1.6 * (center_EB[2] - z_cutting_plane_);
  // z_cut_CB_upper is the upper z_plane position for the CB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_CB_upper =
      std::min(0.5 * (z_cut_CB_lower - inner_radius_C), -0.7 * inner_radius_C);
  // z_cut_EB_upper is the upper z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme.  Note here that 'upper' means 'closer to z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_upper = center_B_[2] - 0.7 * outer_radius_B_;
  // z_cut_EB_lower is the lower z_plane position for the EB endcap,
  // which isn't defined in https://arxiv.org/abs/1206.3015 (because the
  // maps are different).  We choose this plane to make the maps
  // less extreme. Note here that 'lower' means 'farther from z=-infinity'
  // because we are on the -z side of the cutting plane.
  const double z_cut_EB_lower = center_B_[2] + 0.7 * outer_radius_B_;

  // EB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_z_axis),
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_CB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // EB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_z_axis),
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_EB_lower,
          -z_cut_CB_lower, -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // MA Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_z_axis),
      CoordinateMaps::UniformCylindricalFlatEndcap(
          flip_about_xy_plane(center_A_),
          flip_about_xy_plane(center_cutting_plane), outer_radius_A_, radius_MB,
          -z_cut_EA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));
  // MB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(aligned),
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalFlatEndcap(  // LCOV_EXCL_LINE
          center_B_, center_cutting_plane, outer_radius_B_, radius_MB,
          z_cut_EB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CB Filled Cylinder
  add_endcap_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_z_axis),
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cut_CB_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // CB Cylinder
  add_side_to_list_of_maps(
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_z_axis),
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cutting_plane_, -z_cut_CB_upper,
          -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  const size_t ea_endcap_block = block_positions_.at("EAFilledCylinder");
  const size_t ea_side_block = block_positions_.at("EACylinder");
  const size_t ma_endcap_block = block_positions_.at("MAFilledCylinder");
  const size_t eb_endcap_block = block_positions_.at("EBFilledCylinder");
  const size_t eb_side_block = block_positions_.at("EBCylinder");
  const size_t mb_endcap_block = block_positions_.at("MBFilledCylinder");
  const size_t ca_endcap_block = block_positions_.at("CAFilledCylinder");
  const size_t ca_side_block = block_positions_.at("CACylinder");
  const size_t cb_endcap_block = block_positions_.at("CBFilledCylinder");
  const size_t cb_side_block = block_positions_.at("CBCylinder");

  // Excision spheres
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  std::unordered_map<size_t, Direction<3>> abutting_directions_A;
  const size_t inner_shell_A_block = 10;
  size_t inner_shell_B_block = inner_shell_A_block;
  if (include_inner_sphere_A_) {
    // LCOV_EXCL_START
    abutting_directions_A.emplace(inner_shell_A_block,
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP

    // Block numbers of sphereB might depend on whether there is an inner
    // sphereA layer, so increment here to get that right.
    inner_shell_B_block += 1;
  } else {
    abutting_directions_A.emplace(ea_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_A.emplace(ma_endcap_block, Direction<3>::upper_zeta());
    abutting_directions_A.emplace(ea_side_block, Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereA",
      ExcisionSphere<3>{
          radius_A_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_A_)),
          abutting_directions_A});

  std::unordered_map<size_t, Direction<3>> abutting_directions_B;
  if (include_inner_sphere_B_) {
    // LCOV_EXCL_START
    abutting_directions_B.emplace(inner_shell_B_block,
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP
  } else {
    abutting_directions_B.emplace(eb_endcap_block, Direction<3>::upper_zeta());
    abutting_directions_B.emplace(mb_endcap_block, Direction<3>::lower_zeta());
    abutting_directions_B.emplace(eb_side_block, Direction<3>::lower_xi());
  }
  excision_spheres.emplace(
      "ExcisionSphereB",
      ExcisionSphere<3>{
          radius_B_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_B_)),
          abutting_directions_B});

  Domain<3> domain;
  // non-shell maps
  std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_neighbors{
      coordinate_maps.size()};

  // Add a block as a neighbor to a host block
  auto add_block_neighbor =
      [](std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
         const size_t this_block_number, const size_t neighbor_block_number,
         const Direction<3>& direction,
         const OrientationMap<3>& orientation_map,
         const bool are_conforming = true) {
        neighbors[this_block_number].emplace(
            direction,
            BlockNeighbors<3>{{neighbor_block_number},
                              {{neighbor_block_number, orientation_map}},
                              are_conforming});
      };

  // EA Filled Cylinder
  add_block_neighbor(
      inner_neighbors, ea_endcap_block, ea_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(inner_neighbors, ea_endcap_block, ca_endcap_block,
                     Direction<3>::upper_zeta(), aligned);

  // EA Cylinder
  add_block_neighbor(
      inner_neighbors, ea_side_block, ea_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(
      inner_neighbors, ea_side_block, ma_endcap_block,
      Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(inner_neighbors, ea_side_block, ca_side_block,
                     Direction<3>::upper_xi(), aligned);

  // MA Filled Cylinder
  add_block_neighbor(
      inner_neighbors, ma_endcap_block, ea_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(inner_neighbors, ma_endcap_block, mb_endcap_block,
                     Direction<3>::lower_zeta(), aligned);

  // CA Filled Cylinder
  add_block_neighbor(
      inner_neighbors, ca_endcap_block, ca_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(inner_neighbors, ca_endcap_block, ea_endcap_block,
                     Direction<3>::lower_zeta(), aligned);

  // CA Cylinder
  add_block_neighbor(
      inner_neighbors, ca_side_block, ca_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(inner_neighbors, ca_side_block, cb_side_block,
                     Direction<3>::lower_zeta(), aligned);
  add_block_neighbor(inner_neighbors, ca_side_block, ea_side_block,
                     Direction<3>::lower_xi(), aligned);

  // EB Filled Cylinder
  add_block_neighbor(
      inner_neighbors, eb_endcap_block, eb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(inner_neighbors, eb_endcap_block, cb_endcap_block,
                     Direction<3>::lower_zeta(), aligned);

  // EB Cylinder
  add_block_neighbor(
      inner_neighbors, eb_side_block, eb_endcap_block,
      Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(
      inner_neighbors, eb_side_block, mb_endcap_block,
      Direction<3>::upper_zeta(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(inner_neighbors, eb_side_block, cb_side_block,
                     Direction<3>::upper_xi(), aligned);

  // MB Filled Cylinder
  add_block_neighbor(
      inner_neighbors, mb_endcap_block, eb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(inner_neighbors, mb_endcap_block, ma_endcap_block,
                     Direction<3>::upper_zeta(), aligned);

  // CB Filled Cylinder
  add_block_neighbor(
      inner_neighbors, cb_endcap_block, cb_side_block, Direction<3>::upper_xi(),
      OrientationMap<3>{{{Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::lower_xi()}}});
  add_block_neighbor(inner_neighbors, cb_endcap_block, eb_endcap_block,
                     Direction<3>::upper_zeta(), aligned);

  // CB Cylinder
  add_block_neighbor(
      inner_neighbors, cb_side_block, cb_endcap_block,
      Direction<3>::lower_zeta(),
      OrientationMap<3>{{{Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
                          Direction<3>::upper_xi()}}});
  add_block_neighbor(inner_neighbors, cb_side_block, ca_side_block,
                     Direction<3>::upper_zeta(), aligned);
  add_block_neighbor(inner_neighbors, cb_side_block, eb_side_block,
                     Direction<3>::lower_xi(), aligned);

  // Connect the E sphere cylinder blocks to the outermost inner shells and
  // connect the C sphere cylinder blocks to the innermost outer shell
  const OrientationMap<3> upper_shell_to_lower_cyl_endcap{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto lower_cyl_endcap_to_upper_shell =
      upper_shell_to_lower_cyl_endcap.inverse_map();

  const OrientationMap<3> upper_shell_to_upper_cyl_endcap{
      {{Direction<3>::lower_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto upper_cyl_endcap_to_upper_shell =
      upper_shell_to_upper_cyl_endcap.inverse_map();

  const OrientationMap<3> lower_shell_to_upper_cyl_endcap{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto upper_cyl_endcap_to_lower_shell =
      lower_shell_to_upper_cyl_endcap.inverse_map();

  const OrientationMap<3> lower_shell_to_lower_cyl_endcap{
      {{Direction<3>::lower_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto lower_cyl_endcap_to_lower_shell =
      lower_shell_to_lower_cyl_endcap.inverse_map();

  const OrientationMap<3> shell_to_cyl_side{
      {{Direction<3>::upper_xi(), Direction<3>::self(), Direction<3>::self()}}};
  const auto cyl_side_to_shell = shell_to_cyl_side.inverse_map();

  if (include_inner_sphere_A_) {
    // EA Filled Cylinder
    add_block_neighbor(inner_neighbors, ea_endcap_block, inner_shell_A_block,
                       Direction<3>::lower_zeta(),
                       lower_cyl_endcap_to_upper_shell, false);
    // EA Cylinder
    add_block_neighbor(inner_neighbors, ea_side_block, inner_shell_A_block,
                       Direction<3>::lower_xi(), cyl_side_to_shell, false);
    // MA Filled Cylinder
    add_block_neighbor(inner_neighbors, ma_endcap_block, inner_shell_A_block,
                       Direction<3>::upper_zeta(),
                       upper_cyl_endcap_to_upper_shell, false);
  }

  if (include_inner_sphere_B_) {
    // EB Filled Cylinder
    add_block_neighbor(inner_neighbors, eb_endcap_block, inner_shell_B_block,
                       Direction<3>::upper_zeta(),
                       upper_cyl_endcap_to_upper_shell, false);
    // EB Cylinder
    add_block_neighbor(inner_neighbors, eb_side_block, inner_shell_B_block,
                       Direction<3>::lower_xi(), cyl_side_to_shell, false);
    // MB Filled Cylinder
    add_block_neighbor(inner_neighbors, mb_endcap_block, inner_shell_B_block,
                       Direction<3>::lower_zeta(),
                       lower_cyl_endcap_to_upper_shell, false);
  }

  const size_t outer_shell_block = block_positions_.at("OuterShell0");

  // CA Filled Cylinder
  add_block_neighbor(inner_neighbors, ca_endcap_block, outer_shell_block,
                     Direction<3>::upper_zeta(),
                     upper_cyl_endcap_to_lower_shell, false);
  // CA Cylinder
  add_block_neighbor(inner_neighbors, ca_side_block, outer_shell_block,
                     Direction<3>::upper_xi(), cyl_side_to_shell, false);

  // CB Filled Cylinder
  add_block_neighbor(inner_neighbors, cb_endcap_block, outer_shell_block,
                     Direction<3>::lower_zeta(),
                     lower_cyl_endcap_to_lower_shell, false);
  // CB Cylinder
  add_block_neighbor(inner_neighbors, cb_side_block, outer_shell_block,
                     Direction<3>::upper_xi(), cyl_side_to_shell, false);

  // Build blocks in final order.
  std::vector<Block<3>> blocks;
  blocks.reserve(number_of_blocks_);

  // (a) Inner blocks before SH shells.
  for (size_t j = 0; j < inner_shell_A_block; ++j) {
    const std::string& block_name = gsl::at(block_names_, j);
    ASSERT(block_name.find("Cylinder") != std::string::npos,
           "Expected block to be a cylindrical block with the substring "
           "'Cylinder'.");

    const auto cyl_topology = block_name.find("Filled") != std::string::npos ?
        domain::topologies::full_cylinder :
        domain::topologies::cylindrical_shell;
    blocks.emplace_back(std::move(coordinate_maps[j]), j,
                        std::move(inner_neighbors[j]), block_name,
                        cyl_topology);
  }

  auto add_block_id_and_orientation_to_sets =
      [](std::unordered_set<size_t>& ids,
         std::unordered_map<size_t, OrientationMap<3>>& orientations,
         const size_t block_number, const OrientationMap<3>& orientation) {
        ids.insert(block_number);
        orientations.emplace(block_number, orientation);
      };

  using Affine = CoordinateMaps::Affine;
  auto make_spherical_shell_coord_map =
      [](const double inner_radius, const double outer_radius,
         const std::array<double, 3>& aligned_center) {
        CoordinateMaps::Interval radial_map{
            -1.0,
            1.0,
            inner_radius,
            outer_radius,
            ::domain::CoordinateMaps::Distribution::Linear,
            0.0};
        return make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Interval,
                                           CoordinateMaps::Identity<2>>{
                std::move(radial_map), CoordinateMaps::Identity<2>{}},
            CoordinateMaps::SphericalToCartesianPfaffian{},
            CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>{
                Affine{-1.0, 1.0, -1.0 + aligned_center[0],
                       1.0 + aligned_center[0]},
                Affine{-1.0, 1.0, -1.0 + aligned_center[1],
                       1.0 + aligned_center[1]},
                Affine{-1.0, 1.0, -1.0 + aligned_center[2],
                       1.0 + aligned_center[2]}});
      };

  // (b) SH inner shell blocks for InnerSphereA.
  if (include_inner_sphere_A_) {
    // upper_xi → EA endcap, EA side, and MA blocks
    // (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_a_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_a_cyl_orientations;

    // EA Filled Cylinder
    add_block_id_and_orientation_to_sets(
        inner_a_cyl_ids, inner_a_cyl_orientations, ea_endcap_block,
        upper_shell_to_lower_cyl_endcap);
    // EA Cylinder
    add_block_id_and_orientation_to_sets(inner_a_cyl_ids,
                                         inner_a_cyl_orientations,
                                         ea_side_block, shell_to_cyl_side);
    // MA Filled Cylinder
    add_block_id_and_orientation_to_sets(
        inner_a_cyl_ids, inner_a_cyl_orientations, ma_endcap_block,
        upper_shell_to_upper_cyl_endcap);

    auto inner_a_sh_map = make_spherical_shell_coord_map(
        radius_A_, outer_radius_A_, rotate_from_z_to_x_axis(center_A_));

    DirectionMap<3, BlockNeighbors<3>> inner_a_sh_neighbors;
    inner_a_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_a_cyl_ids),
                          std::move(inner_a_cyl_orientations),
                          /*are_conforming=*/false});
    blocks.emplace_back(std::move(inner_a_sh_map), inner_shell_A_block,
                        std::move(inner_a_sh_neighbors),
                        block_names_[inner_shell_A_block],
                        domain::topologies::spherical_shell);
  }

  // (c) SH inner shell blocks for InnerSphereB.
  if (include_inner_sphere_B_) {
    // upper_xi → EB endcap, EB side, and MB blocks
    // (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_b_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_b_cyl_orientations;

    // EB Filled Cylinder
    add_block_id_and_orientation_to_sets(
        inner_b_cyl_ids, inner_b_cyl_orientations, eb_endcap_block,
        upper_shell_to_upper_cyl_endcap);
    // EB Cylinder
    add_block_id_and_orientation_to_sets(inner_b_cyl_ids,
                                         inner_b_cyl_orientations,
                                         eb_side_block, shell_to_cyl_side);
    // MB Filled Cylinder
    add_block_id_and_orientation_to_sets(
        inner_b_cyl_ids, inner_b_cyl_orientations, mb_endcap_block,
        upper_shell_to_lower_cyl_endcap);

    auto inner_b_sh_map = make_spherical_shell_coord_map(
        radius_B_, outer_radius_B_, rotate_from_z_to_x_axis(center_B_));

    DirectionMap<3, BlockNeighbors<3>> inner_b_sh_neighbors;
    inner_b_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_b_cyl_ids),
                          std::move(inner_b_cyl_orientations),
                          /*are_conforming=*/false});

    blocks.emplace_back(std::move(inner_b_sh_map), inner_shell_B_block,
                        std::move(inner_b_sh_neighbors),
                        block_names_[inner_shell_B_block],
                        domain::topologies::spherical_shell);
  }

  // (d) SH outer shell blocks for OuterSphere.
  // lower_xi → all 4 CA, CB blocks (non-conforming, multi-neighbor).
  std::unordered_set<size_t> outer_cyl_ids;
  std::unordered_map<size_t, OrientationMap<3>> outer_cyl_orientations;

  // CA Filled Cylinder
  add_block_id_and_orientation_to_sets(outer_cyl_ids, outer_cyl_orientations,
                                       ca_endcap_block,
                                       lower_shell_to_upper_cyl_endcap);
  // CA Cylinder
  add_block_id_and_orientation_to_sets(outer_cyl_ids, outer_cyl_orientations,
                                       ca_side_block, shell_to_cyl_side);
  // CB Filled Cylinder
  add_block_id_and_orientation_to_sets(outer_cyl_ids, outer_cyl_orientations,
                                       cb_endcap_block,
                                       lower_shell_to_lower_cyl_endcap);
  // CB Cylinder
  add_block_id_and_orientation_to_sets(outer_cyl_ids, outer_cyl_orientations,
                                       cb_side_block, shell_to_cyl_side);

  auto outer_sh_map = make_spherical_shell_coord_map(
      inner_radius_C, outer_radius_, make_array<3>(0.0));

  DirectionMap<3, BlockNeighbors<3>> outer_sh_neighbors;
  outer_sh_neighbors.emplace(
      Direction<3>::lower_xi(),
      BlockNeighbors<3>{std::move(outer_cyl_ids),
                        std::move(outer_cyl_orientations),
                        /*are_conforming=*/false});

  blocks.emplace_back(
      std::move(outer_sh_map), outer_shell_block, std::move(outer_sh_neighbors),
      block_names_[outer_shell_block], domain::topologies::spherical_shell);

  domain =
      Domain<3>{std::move(blocks), std::move(excision_spheres), block_groups_};

  if (time_dependent_options_.has_value()) {
    ASSERT(include_inner_sphere_A_ and include_inner_sphere_B_,
           "When using time dependent maps for the CylindricalBBH domain, you "
           "must include both inner spheres.");
    // Default initialize everything to nullptr so that we only need to set the
    // appropriate block maps for the specific frames
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, 3>>>
        grid_to_distorted_block_maps{number_of_blocks_};
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, 3>>>
        distorted_to_inertial_block_maps{number_of_blocks_};

    // The 0th block always exists and will only need an rigid expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[0] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, true);

    // The first block in the outer shell needs the transition expansion +
    // rotation + translation map from the grid to inertial frame. No maps to
    // the distorted frame
    grid_to_inertial_block_maps[outer_shell_block] =
        time_dependent_options_
            ->grid_to_inertial_map<domain::ObjectLabel::None>(false, false);

    // Inside the excision sphere we add the grid to inertial map from the
    // outer shell. This allows the center of the excisions/horizons to be
    // mapped properly to the inertial frame.
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereA",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true, true));
    domain.inject_time_dependent_map_for_excision_sphere(
        "ExcisionSphereB",
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true, true));

    // The `true` being passed to the functions specifies that the size map
    // *should* be included in the distorted frame.
    grid_to_inertial_block_maps[inner_shell_A_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true);
    grid_to_distorted_block_maps[inner_shell_A_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::A>(
            true);
    distorted_to_inertial_block_maps[inner_shell_A_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::A>(true, true);

    grid_to_inertial_block_maps[inner_shell_B_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true);
    grid_to_distorted_block_maps[inner_shell_B_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::B>(
            true);
    distorted_to_inertial_block_maps[inner_shell_B_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::B>(true, true);

    for (size_t block = 1; block < inner_shell_A_block; ++block) {
      grid_to_inertial_block_maps[block] =
          grid_to_inertial_block_maps[0]->get_clone();
    }

    for (size_t block = 0; block < number_of_blocks_; ++block) {
      domain.inject_time_dependent_map_for_block(
          block, std::move(grid_to_inertial_block_maps[block]),
          std::move(grid_to_distorted_block_maps[block]),
          std::move(distorted_to_inertial_block_maps[block]));
    }
  }

  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
CylindricalBinaryCompactObject::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  const size_t ea_endcap_block = block_positions_.at("EAFilledCylinder");
  const size_t ea_side_block = block_positions_.at("EACylinder");
  const size_t ma_endcap_block = block_positions_.at("MAFilledCylinder");
  const size_t eb_endcap_block = block_positions_.at("EBFilledCylinder");
  const size_t eb_side_block = block_positions_.at("EBCylinder");
  const size_t mb_endcap_block = block_positions_.at("MBFilledCylinder");
  const size_t outer_shell_block = block_positions_.at("OuterShell0");

  if (not include_inner_sphere_A_) {
      // EA Filled Cylinder
      boundary_conditions[ea_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // MA Filled Cylinder
      boundary_conditions[ma_endcap_block][Direction<3>::upper_zeta()] =
          inner_boundary_condition_->get_clone();
      // EA Cylinder
      boundary_conditions[ea_side_block][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
  } else {
    boundary_conditions[block_positions_.at("InnerAShell0")]
                       [Direction<3>::lower_xi()] =
                           inner_boundary_condition_->get_clone();
  }
  if (not include_inner_sphere_B_) {
      // EB Filled Cylinder
      boundary_conditions[eb_endcap_block][Direction<3>::upper_zeta()] =
          inner_boundary_condition_->get_clone();
      // MB Filled Cylinder
      boundary_conditions[mb_endcap_block][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // EB Cylinder
      boundary_conditions[eb_side_block][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
  } else {
    boundary_conditions[block_positions_.at("InnerBShell0")]
                       [Direction<3>::lower_xi()] =
                           inner_boundary_condition_->get_clone();
  }
  boundary_conditions[outer_shell_block][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();

  return boundary_conditions;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>>
CylindricalBinaryCompactObject::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
CylindricalBinaryCompactObject::functions_of_time(
    const std::unordered_map<std::string, double>& initial_expiration_times)
    const {
  return time_dependent_options_.has_value()
             ? time_dependent_options_->create_functions_of_time(
                   initial_expiration_times)
             : std::unordered_map<
                   std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{};
}
}  // namespace domain::creators
