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
#include "Options/ParseError.hpp"

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
    bool include_inner_sphere_B, double outer_radius, bool use_equiangular_map,
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
      use_equiangular_map_(use_equiangular_map),
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

  number_of_blocks_ = 46;
  if (include_inner_sphere_A) {
    number_of_blocks_ += 1;
  }
  if (include_inner_sphere_B) {
    number_of_blocks_ += 1;
  }
  // Outer sphere
  number_of_blocks_ += 1;

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

  // Create block names and groups
  auto add_filled_cylinder_name = [this](const std::string& prefix,
                                         const std::string& group_name) {
    for (const std::string& where :
         {"Center"s, "East"s, "North"s, "West"s, "South"s}) {
      const std::string name =
          std::string(prefix).append("FilledCylinder").append(where);
      block_names_.push_back(name);
      block_groups_[group_name].insert(name);
    }
  };
  auto add_cylinder_name = [this](const std::string& prefix,
                                  const std::string& group_name) {
    for (const std::string& where : {"East"s, "North"s, "West"s, "South"s}) {
      const std::string name =
          std::string(prefix).append("Cylinder").append(where);
      block_names_.push_back(name);
      block_groups_[group_name].insert(name);
    }
  };

  // CA Filled Cylinder
  // 5 blocks: 0 thru 4
  add_filled_cylinder_name("CA", "Outer");

  // CA Cylinder
  // 4 blocks: 5 thru 8
  add_cylinder_name("CA", "Outer");

  // EA Filled Cylinder
  // 5 blocks: 9 thru 13
  add_filled_cylinder_name("EA", "InnerA");

  // EA Cylinder
  // 4 blocks: 14 thru 17
  add_cylinder_name("EA", "InnerA");

  // EB Filled Cylinder
  // 5 blocks: 18 thru 22
  add_filled_cylinder_name("EB", "InnerB");

  // EB Cylinder
  // 4 blocks: 23 thru 26
  add_cylinder_name("EB", "InnerB");

  // MA Filled Cylinder
  // 5 blocks: 27 thru 31
  add_filled_cylinder_name("MA", "InnerA");

  // MB Filled Cylinder
  // 5 blocks: 32 thru 36
  add_filled_cylinder_name("MB", "InnerB");

  // CB Filled Cylinder
  // 5 blocks: 37 thru 41
  add_filled_cylinder_name("CB", "Outer");

  // CB Cylinder
  // 4 blocks: 42 thru 45
  add_cylinder_name("CB", "Outer");

  // Create block names and groups
  auto add_spherical_shell_name = [this](const std::string& prefix,
                                         const std::string& group_name,
                                         const size_t shell_number) {
    const std::string name = std::string(prefix).append("Shell").append(
        std::to_string(shell_number));
    block_names_.push_back(name);
    block_groups_[group_name].insert(name);
  };

  const size_t first_inner_shell_A_block = 46;
  size_t first_inner_shell_B_block = first_inner_shell_A_block;
  first_outer_shell_block = first_inner_shell_A_block;

  if (include_inner_sphere_A) {
    add_spherical_shell_name("InnerA", "InnerSphereA", 0);

    first_inner_shell_B_block += 1;
    first_outer_shell_block += 1;
  }
  if (include_inner_sphere_B) {
    add_spherical_shell_name("InnerB", "InnerSphereB", 0);

    first_outer_shell_block += 1;
  }
  add_spherical_shell_name("Outer", "OuterSphere", 0);

  // Expand initial refinement over all blocks
  const ExpandOverBlocks<std::array<size_t, 3>> expand_over_blocks{
      block_names_, block_groups_};
  try {
    initial_refinement_ = std::visit(
        [this, &expand_over_blocks, &first_inner_shell_A_block,
         &first_inner_shell_B_block]<typename V>(
            const V& v) -> std::vector<std::array<size_t, 3>> {
          if constexpr (std::is_same_v<V, size_t>) {
            std::vector<std::array<size_t, 3>> expanded = expand_over_blocks(v);

            // If one size_t was used to globally define refinement throughout
            // the domain, overwrite the refinement levels to be 0 just for the
            // angular directions of the spherical shells since angular
            // refinement doesn't make since for blocks with spherical
            // harmonics. This allows the user to still easily specify a global
            // refinement level instead of having to manually specify the same
            // refinement for all directions of all block groups with the same
            // number except spherical shell blocks.

            if (include_inner_sphere_A_) {
              // InnerAShell0
              expanded[first_inner_shell_A_block][1] = 0;
              expanded[first_inner_shell_A_block][2] = 0;
            }
            if (include_inner_sphere_B_) {
              // InnerBShell0
              expanded[first_inner_shell_B_block][1] = 0;
              expanded[first_inner_shell_B_block][2] = 0;
            }
            // OuterShell0
            expanded[first_outer_shell_block][1] = 0;
            expanded[first_outer_shell_block][2] = 0;

            return expanded;
          } else {
            (void)first_inner_shell_A_block;
            (void)first_inner_shell_B_block;
            (void)first_outer_shell_block;

            return expand_over_blocks(v);
          }
        },
        initial_refinement);

  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialRefinement': " << error.what());
  }
  if (include_inner_sphere_A_) {
    // validate angular h-refinement == 0 in spherical shell blocks
    if (gsl::at(initial_refinement_, first_inner_shell_A_block)[1] != 0 or
        gsl::at(initial_refinement_, first_inner_shell_A_block)[2] != 0) {
      PARSE_ERROR(context,
                  "Angular h-refinement is not supported for "
                  "spherical-harmonic inner-shell blocks. Specify refinement "
                  "for InnerSphereA as [radial_refinement, 0, 0].");
    }
  }
  if (include_inner_sphere_B_) {
    if (gsl::at(initial_refinement_, first_inner_shell_B_block)[1] != 0 or
        gsl::at(initial_refinement_, first_inner_shell_B_block)[2] != 0) {
      PARSE_ERROR(context,
                  "Angular h-refinement is not supported for "
                  "spherical-harmonic inner-shell blocks. Specify refinement "
                  "for InnerSphereB as [radial_refinement, 0, 0].");
    }
  }
  if (gsl::at(initial_refinement_, first_outer_shell_block)[1] != 0 or
      gsl::at(initial_refinement_, first_outer_shell_block)[2] != 0) {
    PARSE_ERROR(context,
                "Angular h-refinement is not supported for "
                "spherical-harmonic outer-shell blocks. Specify refinement "
                "as [radial_refinement, 0, 0].");
  }

  size_t inner_sphere_A_l_max = 0;
  size_t inner_sphere_A_m_max = 0;
  size_t inner_sphere_B_l_max = 0;
  size_t inner_sphere_B_m_max = 0;
  size_t outer_sphere_l_max = 0;
  size_t outer_sphere_m_max = 0;

  try {
    initial_grid_points_ = std::visit(expand_over_blocks, initial_grid_points);

    inner_sphere_A_l_max = initial_grid_points_[first_inner_shell_A_block][1];
    inner_sphere_A_m_max = initial_grid_points_[first_inner_shell_A_block][2];

    inner_sphere_B_l_max = initial_grid_points_[first_inner_shell_B_block][1];
    inner_sphere_B_m_max = initial_grid_points_[first_inner_shell_B_block][2];

    outer_sphere_l_max =
        initial_grid_points_[first_outer_shell_block][1];
    outer_sphere_m_max = initial_grid_points_[first_outer_shell_block][2];

    // For spherical-harmonic shell blocks, initial_number_of_grid_points_
    // stores {n_radial, l_max, m_max}. Convert (l_max, m_max) to the number of
    // collocation points the spherical-harmonic basis uses in each angular
    // direction.

    // InnerAShell0
    initial_grid_points_[first_inner_shell_A_block][1] =
        ylm::Spherepack::n_theta_points(inner_sphere_A_l_max);
    initial_grid_points_[first_inner_shell_A_block][2] =
        ylm::Spherepack::n_phi_points(inner_sphere_A_m_max);
    // InnerBShell0
    initial_grid_points_[first_inner_shell_B_block][1] =
        ylm::Spherepack::n_theta_points(inner_sphere_B_l_max);
    initial_grid_points_[first_inner_shell_B_block][2] =
        ylm::Spherepack::n_phi_points(inner_sphere_B_m_max);
    // OuterShell0
    initial_grid_points_[first_outer_shell_block][1] =
        ylm::Spherepack::n_theta_points(outer_sphere_l_max);
    initial_grid_points_[first_outer_shell_block][2] =
        ylm::Spherepack::n_phi_points(outer_sphere_m_max);
  } catch (const std::exception& error) {
    PARSE_ERROR(context, "Invalid 'InitialGridPoints': " << error.what());
  }
  // A spherical-harmonic shell is fully specified by a single ell, so
  // l_max must equal m_max.
  if (inner_sphere_A_l_max != inner_sphere_A_m_max) {
    PARSE_ERROR(
        context,
        "Spherical-harmonic inner-shell blocks must have L_max = M_max. "
        "Specify grid points for InnerSphereA as [radial_points, L_max, "
        "L_max].");
  }
  if (inner_sphere_B_l_max != inner_sphere_B_m_max) {
    PARSE_ERROR(
        context,
        "Spherical-harmonic inner-shell blocks must have L_max = M_max. "
        "Specify grid points for InnerSphereB as [radial_points, L_max, "
        "L_max].");
  }
  if (outer_sphere_l_max != outer_sphere_m_max) {
    PARSE_ERROR(
        context,
        "Spherical-harmonic outer-shell blocks must have L_max = M_max. "
        "Specify grid points for OuterSphere as [radial_points, L_max, "
        "L_max].");
  }

  // Now we must change the initial refinement and initial grid points
  // for certain blocks, because the [r, theta, perp] directions do
  // not always correspond to [xi, eta, zeta].  The values in
  // initial_refinement_ must correspond to [xi, eta, zeta].
  //
  // In particular, for cylinders: [xi, eta, zeta] = [r, theta, perp]
  // but for filled cylinders: [xi, eta, zeta] = [perp, theta, r].

  auto swap_refinement_and_grid_points_xi_zeta = [this](const size_t block_id) {
    size_t val = gsl::at(initial_refinement_[block_id], 0);
    gsl::at(initial_refinement_[block_id], 0) =
        gsl::at(initial_refinement_[block_id], 2);
    gsl::at(initial_refinement_[block_id], 2) = val;
    val = gsl::at(initial_grid_points_[block_id], 0);
    gsl::at(initial_grid_points_[block_id], 0) =
        gsl::at(initial_grid_points_[block_id], 2);
    gsl::at(initial_grid_points_[block_id], 2) = val;
  };

  // CA Filled Cylinder
  // 5 blocks: 0 thru 4
  for (size_t block = 0; block < 5; ++block) {
    swap_refinement_and_grid_points_xi_zeta(block);
  }

  // EA Filled Cylinder
  // 5 blocks: 9 thru 13
  for (size_t block = 9; block < 14; ++block) {
    swap_refinement_and_grid_points_xi_zeta(block);
  }

  // EB Filled Cylinder
  // 5 blocks: 18 thru 22
  for (size_t block = 18; block < 23; ++block) {
    swap_refinement_and_grid_points_xi_zeta(block);
  }

  // MA Filled Cylinder
  // 5 blocks: 27 thru 31
  // MB Filled Cylinder
  // 5 blocks: 32 thru 36
  // CB Filled Cylinder
  // 5 blocks: 37 thru 41
  for (size_t block = 27; block < 42; ++block) {
    swap_refinement_and_grid_points_xi_zeta(block);
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

  // Construct vector<CoordMap>s that go from logical coordinates to
  // various blocks making up a unit right cylinder.  These blocks are
  // either the central square blocks, or the surrounding wedge
  // blocks. The radii and bounds are what are expected by the
  // UniformCylindricalEndcap maps, (except cylinder_inner_radius, which
  // determines the internal block boundaries inside the cylinder, and
  // which the UniformCylindricalEndcap maps don't care about).
  const double cylinder_inner_radius = 0.5;
  const double cylinder_outer_radius = 1.0;
  const double cylinder_lower_bound_z = -1.0;
  const double cylinder_upper_bound_z = 1.0;
  const auto logical_to_cylinder_center_maps =
      cyl_wedge_coord_map_center_blocks(
          cylinder_inner_radius, cylinder_lower_bound_z, cylinder_upper_bound_z,
          use_equiangular_map_);
  const auto logical_to_cylinder_surrounding_maps =
      cyl_wedge_coord_map_surrounding_blocks(
          cylinder_inner_radius, cylinder_outer_radius, cylinder_lower_bound_z,
          cylinder_upper_bound_z, use_equiangular_map_, 0.0);

  // Lambda that takes a UniformCylindricalEndcap map and a
  // DiscreteRotation map, composes it with the logical-to-cylinder
  // maps, and adds it to the list of coordinate maps. Also adds
  // boundary conditions if requested.
  auto add_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_center_maps,
       &logical_to_cylinder_surrounding_maps](
          const CoordinateMaps::UniformCylindricalEndcap& endcap_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylinder_center_maps =
            domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_center_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_center_maps.begin()),
            std::make_move_iterator(new_logical_to_cylinder_center_maps.end()));
        auto new_logical_to_cylinder_surrounding_maps =
            domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_surrounding_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.end()));
      };

  // Lambda that takes a UniformCylindricalFlatEndcap map and a
  // DiscreteRotation map, composes it with the logical-to-cylinder
  // maps, and adds it to the list of coordinate maps. Also adds
  // boundary conditions if requested.
  auto add_flat_endcap_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylinder_center_maps,
       &logical_to_cylinder_surrounding_maps](
          const CoordinateMaps::UniformCylindricalFlatEndcap& endcap_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylinder_center_maps =
            domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_center_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_center_maps.begin()),
            std::make_move_iterator(new_logical_to_cylinder_center_maps.end()));
        auto new_logical_to_cylinder_surrounding_maps =
            domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial, 3>(
                logical_to_cylinder_surrounding_maps, endcap_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylinder_surrounding_maps.end()));
      };

  // Construct vector<CoordMap>s that go from logical coordinates to
  // various blocks making up a right cylindrical shell of inner radius 1,
  // outer radius 2, and z-extents from -1 to +1.  These blocks are
  // either the central square blocks, or the surrounding wedge
  // blocks. The radii and bounds are what are expected by the
  // UniformCylindricalEndcap maps.
  const double cylindrical_shell_inner_radius = 1.0;
  const double cylindrical_shell_outer_radius = 2.0;
  const double cylindrical_shell_lower_bound_z = -1.0;
  const double cylindrical_shell_upper_bound_z = 1.0;
  const auto logical_to_cylindrical_shell_maps =
      cyl_wedge_coord_map_surrounding_blocks(
          cylindrical_shell_inner_radius, cylindrical_shell_outer_radius,
          cylindrical_shell_lower_bound_z, cylindrical_shell_upper_bound_z,
          use_equiangular_map_, 1.0);

  // Lambda that takes a UniformCylindricalSide map and a DiscreteRotation
  // map, composes it with the logical-to-cylinder maps, and adds it
  // to the list of coordinate maps.  Also adds boundary conditions if
  // requested.
  auto add_side_to_list_of_maps =
      [&coordinate_maps, &logical_to_cylindrical_shell_maps](
          const CoordinateMaps::UniformCylindricalSide& side_map,
          const CoordinateMaps::DiscreteRotation<3>& rotation_map) {
        auto new_logical_to_cylindrical_shell_maps =
            domain::make_vector_coordinate_map_base<Frame::BlockLogical,
                                                    Frame::Inertial, 3>(
                logical_to_cylindrical_shell_maps, side_map, rotation_map);
        coordinate_maps.insert(
            coordinate_maps.end(),
            std::make_move_iterator(
                new_logical_to_cylindrical_shell_maps.begin()),
            std::make_move_iterator(
                new_logical_to_cylindrical_shell_maps.end()));
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
  // 5 blocks: 0 thru 4
  add_endcap_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(center_EA, make_array<3>(0.0),
                                               radius_EA, inner_radius_C,
                                               z_cut_CA_lower, z_cut_CA_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CA Cylinder
  // 4 blocks: 5 thru 8
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          // codecov complains about the next line being untested.
          // No idea why, since this entire function is called.
          // LCOV_EXCL_START
          center_EA, make_array<3>(0.0), radius_EA, inner_radius_C,
          // LCOV_EXCL_STOP
          z_cut_CA_lower, z_cutting_plane_, z_cut_CA_upper, z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Filled Cylinder
  // 5 blocks: 9 thru 13
  add_endcap_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(center_A_, center_EA,
                                               outer_radius_A_, radius_EA,
                                               z_cut_EA_upper, z_cut_CA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // EA Cylinder
  // 4 blocks: 14 thru 17
  add_side_to_list_of_maps(
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
  // 5 blocks: 18 thru 22
  add_endcap_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_CB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // EB Cylinder
  // 4 blocks: 23 thru 26
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_B_), flip_about_xy_plane(center_EB),
          outer_radius_B_, radius_EB, -z_cut_EB_upper, -z_cut_EB_lower,
          -z_cut_CB_lower, -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // MA Filled Cylinder
  // 5 blocks: 27 thru 31
  add_flat_endcap_to_list_of_maps(
      CoordinateMaps::UniformCylindricalFlatEndcap(
          flip_about_xy_plane(center_A_),
          flip_about_xy_plane(center_cutting_plane), outer_radius_A_, radius_MB,
          -z_cut_EA_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));
  // MB Filled Cylinder
  // 5 blocks: 32 thru 36
  add_flat_endcap_to_list_of_maps(
      // For some reason codecov complains about the next line.
      CoordinateMaps::UniformCylindricalFlatEndcap(  // LCOV_EXCL_LINE
          center_B_, center_cutting_plane, outer_radius_B_, radius_MB,
          z_cut_EB_lower),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_x_axis));

  // CB Filled Cylinder
  // 5 blocks: 37 thru 41
  add_endcap_to_list_of_maps(
      CoordinateMaps::UniformCylindricalEndcap(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cut_CB_upper),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // CB Cylinder
  // 4 blocks: 42 thru 45
  add_side_to_list_of_maps(
      CoordinateMaps::UniformCylindricalSide(
          flip_about_xy_plane(center_EB), make_array<3>(0.0), radius_EB,
          inner_radius_C, -z_cut_CB_lower, -z_cutting_plane_, -z_cut_CB_upper,
          -z_cutting_plane_),
      CoordinateMaps::DiscreteRotation<3>(rotate_to_minus_x_axis));

  // Excision spheres
  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  std::unordered_map<size_t, Direction<3>> abutting_directions_A;
  const size_t first_inner_shell_A_block = 46;
  size_t first_inner_shell_B_block = first_inner_shell_A_block;
  if (include_inner_sphere_A_) {
    // LCOV_EXCL_START
    abutting_directions_A.emplace(first_inner_shell_A_block,
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP

    // Block numbers of sphereB might depend on whether there is an inner
    // sphereA layer, so increment here to get that right.
    first_inner_shell_B_block += 1;
  } else {
    for (size_t i = 0; i < 5; ++i) {
      abutting_directions_A.emplace(9 + i, Direction<3>::lower_zeta());
      abutting_directions_A.emplace(27 + i, Direction<3>::lower_zeta());
    }
    for (size_t i = 0; i < 4; ++i) {
      abutting_directions_A.emplace(14 + i, Direction<3>::lower_xi());
    }
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
    abutting_directions_B.emplace(first_inner_shell_B_block,
                                  Direction<3>::lower_xi());
    // LCOV_EXCL_STOP
  } else {
    for (size_t i = 0; i < 5; ++i) {
      abutting_directions_B.emplace(18 + i, Direction<3>::lower_zeta());
      abutting_directions_B.emplace(32 + i, Direction<3>::lower_zeta());
    }
    for (size_t i = 0; i < 4; ++i) {
      abutting_directions_B.emplace(23 + i, Direction<3>::lower_xi());
    }
  }
  excision_spheres.emplace(
      "ExcisionSphereB",
      ExcisionSphere<3>{
          radius_B_,
          tnsr::I<double, 3, Frame::Grid>(rotate_from_z_to_x_axis(center_B_)),
          abutting_directions_B});

  Domain<3> domain;
  // `coordinate_maps` at this point contains the inner non-shell maps.
  // Total =  num_shells + n_interior_cubes entries
  // We determine auto-topology neighbors for these inner maps
  std::vector<DirectionMap<3, BlockNeighbors<3>>> inner_neighbors;
  set_internal_boundaries<3>(make_not_null(&inner_neighbors), coordinate_maps);

  // Connect the E sphere cylinder blocks to the outermost inner shells and
  // connect the C sphere cylinder blocks to the innermost outer shell
  const OrientationMap<3> shell_to_cyl_endcap_center{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto cyl_endcap_center_to_shell =
      shell_to_cyl_endcap_center.inverse_map();
  const OrientationMap<3> shell_to_cyl_endcap_wedge{
      {{Direction<3>::upper_zeta(), Direction<3>::self(),
        Direction<3>::self()}}};
  const auto cyl_endcap_wedge_to_shell =
      shell_to_cyl_endcap_wedge.inverse_map();
  const OrientationMap<3> shell_to_cyl_side{
      {{Direction<3>::upper_xi(), Direction<3>::self(), Direction<3>::self()}}};
  const auto cyl_side_to_shell = shell_to_cyl_side.inverse_map();

  // Add a shell as a neighor of one of the blocks making up a cylinder
  auto add_cyl_shell_block_neighbor =
      [](std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
         const bool cyl_is_filled, const bool shell_is_outside_cyl,
         const size_t cyl_block_number, const size_t shell_block_number,
         const OrientationMap<3>& orientation_map) {
        const Direction<3> direction =
            cyl_is_filled ? (shell_is_outside_cyl ? Direction<3>::upper_zeta()
                                                  : Direction<3>::lower_zeta())
                          : (shell_is_outside_cyl ? Direction<3>::upper_xi()
                                                  : Direction<3>::lower_xi());

        neighbors[cyl_block_number].emplace(
            direction,
            BlockNeighbors<3>{{shell_block_number},
                              {{shell_block_number, orientation_map}},
                              /*are_conforming=*/false});
      };

  // Add a shell as a neighor of each of the blocks making up a cylindrical
  // endcap
  auto add_cyl_endcap_shell_neighbors =
      [&add_cyl_shell_block_neighbor, &cyl_endcap_center_to_shell,
       &cyl_endcap_wedge_to_shell](
          std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
          const bool shell_is_outside_cyl, const size_t first_cyl_block_number,
          const size_t shell_block_number) {
        const bool cyl_is_filled = true;
        add_cyl_shell_block_neighbor(neighbors, cyl_is_filled,
                                     shell_is_outside_cyl,
                                     first_cyl_block_number, shell_block_number,
                                     cyl_endcap_center_to_shell);
        for (size_t j = first_cyl_block_number + 1;
             j < first_cyl_block_number + 5; j++) {
          add_cyl_shell_block_neighbor(
              neighbors, cyl_is_filled, shell_is_outside_cyl, j,
              shell_block_number, cyl_endcap_wedge_to_shell);
        }
      };

  // Add a shell as a neighor of each of the blocks making up a cylindrical side
  auto add_cyl_side_shell_neighbors =
      [&add_cyl_shell_block_neighbor, &cyl_side_to_shell](
          std::vector<DirectionMap<3, BlockNeighbors<3>>>& neighbors,
          const bool shell_is_outside_cyl, const size_t first_cyl_block_number,
          const size_t shell_block_number) {
        const bool cyl_is_filled = false;
        for (size_t j = first_cyl_block_number; j < first_cyl_block_number + 4;
             j++) {
          add_cyl_shell_block_neighbor(neighbors, cyl_is_filled,
                                       shell_is_outside_cyl, j,
                                       shell_block_number, cyl_side_to_shell);
        }
      };

  const size_t first_ea_endcap_block = 9;
  const size_t first_ea_side_block = 14;
  const size_t first_ma_endcap_block = 27;

  if (include_inner_sphere_A_) {
    // EA Filled Cylinder
    add_cyl_endcap_shell_neighbors(inner_neighbors, false,
                                   first_ea_endcap_block,
                                   first_inner_shell_A_block);
    // EA Cylinder
    add_cyl_side_shell_neighbors(inner_neighbors, false, first_ea_side_block,
                                 first_inner_shell_A_block);
    // MA Filled Cylinder
    add_cyl_endcap_shell_neighbors(inner_neighbors, false,
                                   first_ma_endcap_block,
                                   first_inner_shell_A_block);
  }

  const size_t first_eb_endcap_block = 18;
  const size_t first_eb_side_block = 23;
  const size_t first_mb_endcap_block = 32;

  if (include_inner_sphere_B_) {
    // EB Filled Cylinder
    add_cyl_endcap_shell_neighbors(inner_neighbors, false,
                                   first_eb_endcap_block,
                                   first_inner_shell_B_block);
    // EB Cylinder
    add_cyl_side_shell_neighbors(inner_neighbors, false, first_eb_side_block,
                                 first_inner_shell_B_block);
    // MB Filled Cylinder
    add_cyl_endcap_shell_neighbors(inner_neighbors, false,
                                   first_mb_endcap_block,
                                   first_inner_shell_B_block);
  }

  const size_t first_ca_endcap_block = 0;
  const size_t first_ca_side_block = 5;
  const size_t first_cb_endcap_block = 37;
  const size_t first_cb_side_block = 42;

  // CA Filled Cylinder
  add_cyl_endcap_shell_neighbors(inner_neighbors, true, first_ca_endcap_block,
                                 first_outer_shell_block);
  // CA Cylinder
  add_cyl_side_shell_neighbors(inner_neighbors, true, first_ca_side_block,
                               first_outer_shell_block);
  // CB Filled Cylinder
  add_cyl_endcap_shell_neighbors(inner_neighbors, true, first_cb_endcap_block,
                                 first_outer_shell_block);
  // CB Cylinder
  add_cyl_side_shell_neighbors(inner_neighbors, true, first_cb_side_block,
                               first_outer_shell_block);

  // Build blocks in final order.
  std::vector<Block<3>> blocks;
  blocks.reserve(number_of_blocks_);

  // (a) Inner blocks before SH shells.
  for (size_t j = 0; j < first_inner_shell_A_block; ++j) {
    blocks.emplace_back(std::move(coordinate_maps[j]), j,
                        std::move(inner_neighbors[j]), block_names_[j]);
  }

  // Add one of the blocks making up a cylindeical endcap as a neighbor of a
  // shell
  auto add_shell_cyl_endcap_neighbors =
      [&shell_to_cyl_endcap_center, &shell_to_cyl_endcap_wedge](
          std::unordered_set<size_t>& cyl_ids,
          std::unordered_map<size_t, OrientationMap<3>>& cyl_orientations,
          const size_t first_cyl_block_number) {
        cyl_ids.insert(first_cyl_block_number);
        cyl_orientations.emplace(first_cyl_block_number,
                                 shell_to_cyl_endcap_center);
        for (size_t j = first_cyl_block_number + 1;
             j < first_cyl_block_number + 5; ++j) {
          cyl_ids.insert(j);
          cyl_orientations.emplace(j, shell_to_cyl_endcap_wedge);
        }
      };

  // Add one of the blocks making up a cylindrical side as a neighbor of a shell
  auto add_shell_cyl_side_neighbors =
      [&shell_to_cyl_side](
          std::unordered_set<size_t>& cyl_ids,
          std::unordered_map<size_t, OrientationMap<3>>& cyl_orientations,
          const size_t first_cyl_block_number) {
        for (size_t j = first_cyl_block_number; j < first_cyl_block_number + 4;
             ++j) {
          cyl_ids.insert(j);
          cyl_orientations.emplace(j, shell_to_cyl_side);
        }
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
    // lower_xi → all 18 CA, CB blocks (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_a_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_a_cyl_orientations;

    // EA Filled Cylinder
    add_shell_cyl_endcap_neighbors(inner_a_cyl_ids, inner_a_cyl_orientations,
                                   first_ea_endcap_block);
    // EA Cylinder
    add_shell_cyl_side_neighbors(inner_a_cyl_ids, inner_a_cyl_orientations,
                                 first_ea_side_block);
    // MA Filled Cylinder
    add_shell_cyl_endcap_neighbors(inner_a_cyl_ids, inner_a_cyl_orientations,
                                   first_ma_endcap_block);

    auto inner_a_sh_map = make_spherical_shell_coord_map(
        radius_A_, outer_radius_A_, rotate_from_z_to_x_axis(center_A_));

    DirectionMap<3, BlockNeighbors<3>> inner_a_sh_neighbors;
    inner_a_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_a_cyl_ids),
                          std::move(inner_a_cyl_orientations),
                          /*are_conforming=*/false});
    blocks.emplace_back(std::move(inner_a_sh_map), first_inner_shell_A_block,
                        std::move(inner_a_sh_neighbors),
                        block_names_[first_inner_shell_A_block],
                        domain::topologies::spherical_shell);
  }

  // (c) SH inner shell blocks for InnerSphereB.
  if (include_inner_sphere_B_) {
    // lower_xi → all 18 CA, CB blocks (non-conforming, multi-neighbor).
    std::unordered_set<size_t> inner_b_cyl_ids;
    std::unordered_map<size_t, OrientationMap<3>> inner_b_cyl_orientations;

    // EB Filled Cylinder
    add_shell_cyl_endcap_neighbors(inner_b_cyl_ids, inner_b_cyl_orientations,
                                   first_eb_endcap_block);
    // EB Cylinder
    add_shell_cyl_side_neighbors(inner_b_cyl_ids, inner_b_cyl_orientations,
                                 first_eb_side_block);
    // MB Filled Cylinder
    add_shell_cyl_endcap_neighbors(inner_b_cyl_ids, inner_b_cyl_orientations,
                                   first_mb_endcap_block);

    auto inner_b_sh_map = make_spherical_shell_coord_map(
        radius_B_, outer_radius_B_, rotate_from_z_to_x_axis(center_B_));

    DirectionMap<3, BlockNeighbors<3>> inner_b_sh_neighbors;
    inner_b_sh_neighbors.emplace(
        Direction<3>::upper_xi(),
        BlockNeighbors<3>{std::move(inner_b_cyl_ids),
                          std::move(inner_b_cyl_orientations),
                          /*are_conforming=*/false});

    blocks.emplace_back(std::move(inner_b_sh_map), first_inner_shell_B_block,
                        std::move(inner_b_sh_neighbors),
                        block_names_[first_inner_shell_B_block],
                        domain::topologies::spherical_shell);
  }

  // (d) SH outer shell blocks for OuterSphere.
  // lower_xi → all 18 CA, CB blocks (non-conforming, multi-neighbor).
  std::unordered_set<size_t> outer_cyl_ids;
  std::unordered_map<size_t, OrientationMap<3>> outer_cyl_orientations;

  // CA Filled Cylinder
  add_shell_cyl_endcap_neighbors(outer_cyl_ids, outer_cyl_orientations,
                                 first_ca_endcap_block);
  // CA Cylinder
  add_shell_cyl_side_neighbors(outer_cyl_ids, outer_cyl_orientations,
                               first_ca_side_block);
  // CB Filled Cylinder
  add_shell_cyl_endcap_neighbors(outer_cyl_ids, outer_cyl_orientations,
                                 first_cb_endcap_block);
  // CB Cylinder
  add_shell_cyl_side_neighbors(outer_cyl_ids, outer_cyl_orientations,
                               first_cb_side_block);

  auto outer_sh_map = make_spherical_shell_coord_map(
      inner_radius_C, outer_radius_,
      rotate_from_z_to_x_axis(center_cutting_plane));

  DirectionMap<3, BlockNeighbors<3>> outer_sh_neighbors;
  outer_sh_neighbors.emplace(
      Direction<3>::lower_xi(),
      BlockNeighbors<3>{std::move(outer_cyl_ids),
                        std::move(outer_cyl_orientations),
                        /*are_conforming=*/false});

  blocks.emplace_back(std::move(outer_sh_map), first_outer_shell_block,
                      std::move(outer_sh_neighbors),
                      block_names_[first_outer_shell_block],
                      domain::topologies::spherical_shell);

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
    grid_to_inertial_block_maps[first_outer_shell_block] =
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
    grid_to_inertial_block_maps[first_inner_shell_A_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::A>(
            true, true);
    grid_to_distorted_block_maps[first_inner_shell_A_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::A>(
            true);
    distorted_to_inertial_block_maps[first_inner_shell_A_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::A>(true, true);

    grid_to_inertial_block_maps[first_inner_shell_B_block] =
        time_dependent_options_->grid_to_inertial_map<domain::ObjectLabel::B>(
            true, true);
    grid_to_distorted_block_maps[first_inner_shell_B_block] =
        time_dependent_options_->grid_to_distorted_map<domain::ObjectLabel::B>(
            true);
    distorted_to_inertial_block_maps[first_inner_shell_B_block] =
        time_dependent_options_
            ->distorted_to_inertial_map<domain::ObjectLabel::B>(true, true);

    for (size_t block = 1; block < number_of_blocks_; ++block) {
      if (block == first_inner_shell_A_block or
          block == first_inner_shell_B_block or
          block == first_outer_shell_block) {
        continue;  // Already initialized
      } else if (block > first_inner_shell_A_block and
                 block < first_inner_shell_B_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_inner_shell_A_block]->get_clone();
        if (grid_to_distorted_block_maps[first_inner_shell_A_block] !=
            nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[first_inner_shell_A_block]
                  ->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[first_inner_shell_A_block]
                  ->get_clone();
        }
      } else if (block > first_inner_shell_B_block and
                 block < first_outer_shell_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_inner_shell_B_block]->get_clone();
        if (grid_to_distorted_block_maps[first_inner_shell_B_block] !=
            nullptr) {
          grid_to_distorted_block_maps[block] =
              grid_to_distorted_block_maps[first_inner_shell_B_block]
                  ->get_clone();
          distorted_to_inertial_block_maps[block] =
              distorted_to_inertial_block_maps[first_inner_shell_B_block]
                  ->get_clone();
        }
      } else if (block > first_outer_shell_block) {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[first_outer_shell_block]->get_clone();
      } else {
        grid_to_inertial_block_maps[block] =
            grid_to_inertial_block_maps[0]->get_clone();
      }
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
  for (size_t i = 0; i < 5; ++i) {
    if (not include_inner_sphere_A_) {
      // EA Filled Cylinder
      boundary_conditions[i + 9][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // MA Filled Cylinder
      boundary_conditions[i + 27][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
    }
    if (not include_inner_sphere_B_) {
      // EB Filled Cylinder
      boundary_conditions[i + 18][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
      // MB Filled Cylinder
      boundary_conditions[i + 32][Direction<3>::lower_zeta()] =
          inner_boundary_condition_->get_clone();
    }
  }
  for (size_t i = 0; i < 4; ++i) {
    if (not include_inner_sphere_A_) {
      // EA Cylinder
      boundary_conditions[i + 14][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
    }
    if (not include_inner_sphere_B_) {
      // EB Cylinder
      boundary_conditions[i + 23][Direction<3>::lower_xi()] =
          inner_boundary_condition_->get_clone();
    }
  }

  size_t last_block = 46;
  if (include_inner_sphere_A_) {
    boundary_conditions[last_block][Direction<3>::lower_xi()] =
        inner_boundary_condition_->get_clone();
    last_block += 1;
  }
  if (include_inner_sphere_B_) {
    boundary_conditions[last_block][Direction<3>::lower_xi()] =
        inner_boundary_condition_->get_clone();
    last_block += 1;
  }
  boundary_conditions[last_block][Direction<3>::upper_xi()] =
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
