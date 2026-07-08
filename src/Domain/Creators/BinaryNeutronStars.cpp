// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/BinaryNeutronStars.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/CylindricalFlatEndcapInterior.hpp"
#include "Domain/CoordinateMaps/CylindricalSphericalShell.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/CoordinateMaps/FlatOffsetWedge.hpp"
#include "Domain/CoordinateMaps/Interval.hpp"
#include "Domain/CoordinateMaps/PolarToCartesian.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/SphericalToCartesianPfaffian.hpp"
#include "Domain/CoordinateMaps/UniformCylindricalSide.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/ShellDistribution.hpp"
#include "Domain/Creators/TimeDependentOptions/BinaryCompactObject.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Domain/ExcisionSphere.hpp"
#include "Domain/Structure/BlockNeighbors.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Topology.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::creators {
namespace {
// The x-coordinate on the sphere of radius `radius` about the origin that is
// an equal arc length from the slices at x = `lower` and x = `upper`.
double equal_arc_midpoint(const double lower, const double upper,
                          const double radius) {
  return radius * sin(0.5 * (asin(upper / radius) + asin(lower / radius)));
}

// With Bulge=true, the x-coordinate on the sphere of radius
// `cylinder_outer_radius` about the origin where the filled cylinder at the
// end x = `end_x` of the cubed cylinders meets the hollow cylinders. This is
// the radial projection from the origin of the circle where the plane
// x = `end_x` cuts the deformed-cube wedges' outer sphere, which has radius
// `wedge_sphere_radius` and is centered a distance `distance_to_center` from
// that plane.
double bulged_sphere_extent(const double end_x, const double distance_to_center,
                            const double wedge_sphere_radius,
                            const double cylinder_outer_radius) {
  return std::copysign(
      cylinder_outer_radius / sqrt(1.0 + (square(wedge_sphere_radius) -
                                          square(distance_to_center)) /
                                             square(end_x)),
      end_x);
}

// Expands an option that is either a single value for all four cubed-cylinder
// groups or one value per group.
std::array<size_t, 4> per_group(
    const std::variant<size_t, std::array<size_t, 4>>& option) {
  if (std::holds_alternative<size_t>(option)) {
    return make_array<4>(std::get<size_t>(option));
  }
  return std::get<std::array<size_t, 4>>(option);
}
}  // namespace

BinaryNeutronStars::BinaryNeutronStars(
    const std::array<double, 4>& x_axis_points, double wedge_inner_radius,
    double wedge_outer_radius, double cylinder_outer_radius,
    double outer_radius,
    const BinaryNeutronStars::CubeInitialGridPoints::type& cube_grid_points,
    const BinaryNeutronStars::CubeInitialXRefinement::type& cube_x_refinement,
    const BinaryNeutronStars::CubeInitialYZRefinement::type& cube_yz_refinement,
    size_t wedge_prism_radial_grid_points, size_t wedge_prism_radial_refinement,
    size_t cylinder_radial_grid_points, size_t cylinder_radial_refinement,
    size_t b2_angular_grid_points,
    std::array<size_t, 2> hollow_cylinder_angular_grid_points,
    const BinaryNeutronStars::SphericalShellsInitialRadialRefinement::type&
        spherical_shells_radial_refinement,
    size_t spherical_shells_radial_grid_points, size_t spherical_harmonic_l,
    std::vector<double> spherical_shells_radial_partitioning,
    const BinaryNeutronStars::SphericalShellsRadialDistribution::type&
        spherical_shells_radial_distribution,
    bool bulge,
    std::optional<bco::TimeDependentMapOptions<true>> time_dependent_options,
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        outer_boundary_condition,
    const Options::Context& context)
    : leftmost_x_(x_axis_points[0]),
      center_B_(x_axis_points[1]),
      center_A_(x_axis_points[2]),
      rightmost_x_(x_axis_points[3]),
      wedge_inner_radius_(wedge_inner_radius),
      wedge_outer_radius_(wedge_outer_radius),
      cylinder_outer_radius_(cylinder_outer_radius),
      outer_radius_(outer_radius),
      cube_grid_points_(per_group(cube_grid_points)),
      cube_x_refinement_(per_group(cube_x_refinement)),
      cube_yz_refinement_(per_group(cube_yz_refinement)),
      wedge_prism_radial_grid_points_(wedge_prism_radial_grid_points),
      wedge_prism_radial_refinement_(wedge_prism_radial_refinement),
      cylinder_radial_grid_points_(cylinder_radial_grid_points),
      cylinder_radial_refinement_(cylinder_radial_refinement),
      b2_angular_grid_points_(b2_angular_grid_points),
      hollow_cylinder_angular_grid_points_(hollow_cylinder_angular_grid_points),
      spherical_shells_radial_grid_points_(spherical_shells_radial_grid_points),
      spherical_harmonic_l_(spherical_harmonic_l),
      spherical_shells_radial_partitioning_(
          std::move(spherical_shells_radial_partitioning)),
      bulge_(bulge),
      outer_boundary_condition_(std::move(outer_boundary_condition)),
      time_dependent_options_(std::move(time_dependent_options)) {
  if (leftmost_x_ >= center_B_ or center_B_ >= 0.0 or 0.0 >= center_A_ or
      center_A_ >= rightmost_x_) {
    PARSE_ERROR(context,
                "XAxisPoints [LeftEnd, CenterB, CenterA, RightEnd] must "
                "satisfy LeftEnd < CenterB < 0 < CenterA < RightEnd, but got "
                    << "[" << leftmost_x_ << ", " << center_B_ << ", "
                    << center_A_ << ", " << rightmost_x_ << "].");
  }
  if (center_A_ > std::abs(center_B_)) {
    PARSE_ERROR(context,
                "XAxisPoints [LeftEnd, CenterB, CenterA, RightEnd] must "
                "satisfy CenterA <= |CenterB|, but got CenterA="
                    << center_A_ << " > |CenterB|=" << std::abs(center_B_)
                    << ". Increase |CenterB| or decrease CenterA.");
  }
  if (wedge_inner_radius_ <= 0.0) {
    PARSE_ERROR(context, "WedgeInnerRadius must be positive, got "
                             << wedge_inner_radius_ << ".");
  }
  if (wedge_outer_radius_ <= 0.0) {
    PARSE_ERROR(context, "WedgeOuterRadius must be positive, got "
                             << wedge_outer_radius_ << ".");
  }
  if (outer_radius_ <= cylinder_outer_radius_) {
    PARSE_ERROR(
        context,
        "OuterRadius must be greater than CylinderOuterRadius, got "
            << outer_radius_ << " <= " << cylinder_outer_radius_
            << ". Increase OuterRadius or decrease CylinderOuterRadius.");
  }
  if (cylinder_outer_radius_ <= wedge_outer_radius_) {
    PARSE_ERROR(
        context,
        "CylinderOuterRadius must be greater than WedgeOuterRadius, got "
            << cylinder_outer_radius_ << " <= " << wedge_outer_radius_
            << ". Increase CylinderOuterRadius or decrease WedgeOuterRadius.");
  }
  if (wedge_outer_radius_ <= wedge_inner_radius_) {
    PARSE_ERROR(context,
                "WedgeOuterRadius must be greater than WedgeInnerRadius, got "
                    << wedge_outer_radius_ << " <= " << wedge_inner_radius_
                    << ". Increase WedgeOuterRadius or decrease "
                       "WedgeInnerRadius.");
  }
  // Check CylindricalFlatEndcapInterior validity condition t_sphere > 1.
  // With proj_center = center_two = origin, t_sphere = R2/sqrt(w^2 + c1z^2)
  // where c1z is the x-coordinate of the flat disk center (leftmost_x_ on the
  // B side, rightmost_x_ on the A side).  t_sphere > 1 requires
  // R2^2 > w^2 + c1z^2. Check whichever side has the larger |c1z|. With
  // Bulge=false this also ensures that the far corners of the cubed cylinders
  // lie inside CylinderOuterRadius.
  const double c1z_max_sq =
      std::max(leftmost_x_ * leftmost_x_, rightmost_x_ * rightmost_x_);
  if (cylinder_outer_radius_ * cylinder_outer_radius_ <=
      wedge_outer_radius_ * wedge_outer_radius_ + c1z_max_sq) {
    PARSE_ERROR(
        context,
        "CylindricalFlatEndcapInterior (the B2s' map) requires "
        "CylinderOuterRadius^2 > WedgeOuterRadius^2 + max(LeftEnd^2, "
        "RightEnd^2), but got "
            << cylinder_outer_radius_ * cylinder_outer_radius_
            << " <= " << wedge_outer_radius_ * wedge_outer_radius_ + c1z_max_sq
            << " (CylinderOuterRadius=" << cylinder_outer_radius_
            << ", WedgeOuterRadius=" << wedge_outer_radius_
            << ", max|x|=" << sqrt(c1z_max_sq)
            << "). Increase CylinderOuterRadius or bring LeftEnd/RightEnd "
               "closer to zero (decrease |LeftEnd| and |RightEnd|).");
  }
  // CylindricalFlatEndcapInterior requires radius_two/radius_one in [1/10,10].
  if (cylinder_outer_radius_ > 10.0 * wedge_outer_radius_) {
    PARSE_ERROR(context,
                "CylindricalFlatEndcapInterior radius ratio condition "
                "violated: CylinderOuterRadius > 10 * WedgeOuterRadius. "
                "Increase WedgeOuterRadius or decrease CylinderOuterRadius.");
  }
  // CylindricalFlatEndcapInterior has only been tested for certain values
  for (const auto& [name, end_x] : {std::pair{"LeftEnd", leftmost_x_},
                                    std::pair{"RightEnd", rightmost_x_}}) {
    if (std::abs(end_x) < 0.05 * cylinder_outer_radius_ or
        std::abs(end_x) > 0.95 * cylinder_outer_radius_) {
      PARSE_ERROR(context,
                  "CylindricalFlatEndcapInterior (the B2s' map) requires |"
                      << name << "| to be between 0.05 and 0.95 times "
                      << "CylinderOuterRadius, but got |" << name
                      << "|=" << std::abs(end_x)
                      << " and CylinderOuterRadius=" << cylinder_outer_radius_
                      << ". Adjust " << name << " or CylinderOuterRadius.");
    }
  }
  using domain::BoundaryConditions::is_periodic;
  if (is_periodic(outer_boundary_condition_)) {
    PARSE_ERROR(
        context,
        "Cannot have periodic boundary conditions with a binary domain");
  }

  bulge_radius_A_ = wedge_outer_radius_;
  bulge_radius_B_ = wedge_outer_radius_;
  if (bulge_) {
    // Compute effective radii for the two FOW sphere groups (they must be
    // different radii if center_A_ != center_B_ for their slices to match at
    // x=0)
    const double r_inner = wedge_inner_radius_;
    const double R_B = wedge_outer_radius_;
    // FOW constraint D^2 + 2L^2 < R^2; L = r_inner/sqrt(2), so
    // D^2 + r_inner^2 < R^2.
    // BRight: D = |cB|  ->  cB^2 + r^2 < R_B^2. This must be checked before
    // computing R_A, because it guarantees that R_B^2 - (cB^2 - cA^2) > 0.
    // It is also equivalent to the ALeft constraint cA^2 + r^2 < R_A^2.
    if (center_B_ * center_B_ + r_inner * r_inner >= R_B * R_B) {
      PARSE_ERROR(
          context,
          "Bulge=true FlatOffsetWedge BRight constraint "
          "cB^2 + r^2 < R_B^2 violated: "
              << center_B_ * center_B_ + r_inner * r_inner
              << " >= " << R_B * R_B
              << ". Decrease WedgeInnerRadius or the star separations, or "
                 "increase WedgeOuterRadius.");
    }
    // Since CenterA <= |CenterB|, the sphere around A is the one whose
    // radius is adjusted.
    const double R_A =
        sqrt(square(R_B) - square(center_B_) + square(center_A_));
    bulge_radius_B_ = R_B;
    bulge_radius_A_ = R_A;
    // BLeft: D = center_B_ - leftmost_x_
    const double d_bleft = center_B_ - leftmost_x_;  // > 0
    if (d_bleft * d_bleft + r_inner * r_inner >= R_B * R_B) {
      PARSE_ERROR(context,
                  "Bulge=true FlatOffsetWedge BLeft constraint "
                  "D_BLeft^2 + r^2 < R_B^2 violated: "
                      << d_bleft * d_bleft + r_inner * r_inner
                      << " >= R_B^2 = " << R_B * R_B
                      << ". Decrease WedgeInnerRadius, bring LeftEnd closer "
                         "to CenterB, or increase WedgeOuterRadius.");
    }
    // ARight: D = rightmost_x_ - center_A_
    const double d_aright = rightmost_x_ - center_A_;
    if (d_aright * d_aright + r_inner * r_inner >= R_A * R_A) {
      PARSE_ERROR(context,
                  "Bulge=true FlatOffsetWedge ARight constraint "
                  "D_ARight^2 + r^2 < R_A^2 violated: "
                      << d_aright * d_aright + r_inner * r_inner
                      << " >= R_A^2 = " << R_A * R_A
                      << ". Decrease WedgeInnerRadius, bring RightEnd closer "
                         "to CenterA, or increase WedgeOuterRadius.");
    }
    // FlatOffsetWedge has only been tested for certain values
    {
      // Same value as in the FlatOffsetWedge constructor
      constexpr double epsilon = 0.1;
      const double half_width = r_inner / std::numbers::sqrt2;
      const std::array<std::tuple<const char*, double, double>, 4> wedges{
          {{"BLeft", d_bleft, R_B},
           {"BRight", -center_B_, R_B},
           {"ALeft", center_A_, R_A},
           {"ARight", d_aright, R_A}}};
      for (const auto& [group, length, radius] : wedges) {
        if (length < epsilon * radius or length > (1.0 - epsilon) * radius) {
          PARSE_ERROR(context,
                      "Bulge=true FlatOffsetWedge "
                          << group << " constraint violated: the length of the "
                          << group << " cubed cylinder along x (" << length
                          << ") must be between " << epsilon << " and "
                          << 1.0 - epsilon
                          << " times the radius of the sphere on which the "
                             "outer faces of its wedges lie ("
                          << radius << "). Adjust XAxisPoints or "
                          << "WedgeOuterRadius.");
        }
        if (half_width < epsilon * radius) {
          PARSE_ERROR(context,
                      "Bulge=true FlatOffsetWedge "
                          << group
                          << " constraint violated: WedgeInnerRadius/sqrt(2) ("
                          << half_width << ") must be at least " << epsilon
                          << " times the radius of the sphere on which the "
                             "outer faces of the "
                          << group << " wedges lie (" << radius
                          << "). Increase WedgeInnerRadius or decrease "
                             "WedgeOuterRadius.");
        }
        if (square(1.0 - epsilon) * (square(radius) - square(length)) <
            square(r_inner)) {
          PARSE_ERROR(
              context,
              "Bulge=true FlatOffsetWedge "
                  << group << " constraint violated: WedgeInnerRadius^2 ("
                  << square(r_inner) << ") must be at most "
                  << square(1.0 - epsilon) << " (R^2 - D^2) = "
                  << square(1.0 - epsilon) * (square(radius) - square(length))
                  << ", where R (" << radius
                  << ") is the radius of the sphere on which the outer faces "
                     "of the wedges lie and D ("
                  << length << ") is the length of the " << group
                  << " cubed cylinder along x. Decrease WedgeInnerRadius or "
                     "increase WedgeOuterRadius.");
        }
      }
    }
    // UniformCylindricalSide requires the upper cut of the inner sphere to
    // subtend a polar angle < 72 deg from its north pole, i.e.
    // (sphere_center_to_cut_plane) / R > cos(72 deg) ~ 0.309.
    // For BLeft the cut height is d_bleft = center_B_ - leftmost_x_;
    // for ARight the cut height is d_aright = rightmost_x_ - center_A_.
    if (d_bleft / R_B <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(context,
                  "Bulge=true UniformCylindricalSide BLeft constraint "
                  "(CenterB - LeftEnd)/R_B > cos(72 deg) ~ 0.309 violated: "
                      << d_bleft << "/" << R_B << " = " << d_bleft / R_B
                      << ". Decrease LeftEnd (move it farther from CenterB), "
                         "or decrease WedgeOuterRadius.");
    }
    if (d_aright / R_A <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(context,
                  "Bulge=true UniformCylindricalSide ARight constraint "
                  "(RightEnd - CenterA)/R_A > cos(72 deg) ~ 0.309 violated: "
                      << d_aright << "/" << R_A << " = " << d_aright / R_A
                      << ". Increase RightEnd (move it farther from CenterA), "
                         "or decrease WedgeOuterRadius.");
    }
    const auto far_corner_of_bulge = [r_inner](const double distance_to_center,
                                               const double r_outer) {
      return sqrt(square(distance_to_center + r_inner) + square(r_outer) -
                  square(r_inner));
    };
    const double far_corner_aleft = far_corner_of_bulge(center_A_, R_A);
    const double far_corner_bright =
        far_corner_of_bulge(std::abs(center_B_), R_B);
    const double far_corner_bleft = far_corner_of_bulge(d_bleft, R_B);
    const double far_corner_aright = far_corner_of_bulge(d_aright, R_A);
    if (far_corner_aleft >= cylinder_outer_radius_ or
        far_corner_bright >= cylinder_outer_radius_ or
        far_corner_bleft >= cylinder_outer_radius_ or
        far_corner_aright >= cylinder_outer_radius_) {
      PARSE_ERROR(
          context,
          "Bulge=true: the far corner of a FlatOffsetWedge block lies outside "
          "the outer cylinder (CylinderOuterRadius="
              << cylinder_outer_radius_ << "). Far-corner radii: ALeft="
              << far_corner_aleft << ", BRight=" << far_corner_bright
              << ", BLeft=" << far_corner_bleft
              << ", ARight=" << far_corner_aright
              << ". Decrease WedgeOuterRadius or WedgeInnerRadius, bring "
                 "LeftEnd/RightEnd closer to zero (decrease their "
                 "magnitude), or increase CylinderOuterRadius.");
    }
    // UniformCylindricalSide also requires the upper cut of the outer sphere
    // to subtend a polar angle < 72 deg from its north pole. For the BRight and
    // ALeft hollow cylinders this cut is at the equal-arc midpoint between x=0
    // and where the filled cylinder meets the hollow cylinders, so we need
    // |midpoint| / CylinderOuterRadius > cos(72 deg) ~ 0.309.
    const double b_midpoint = equal_arc_midpoint(
        bulged_sphere_extent(leftmost_x_, d_bleft, R_B, cylinder_outer_radius_),
        0.0, cylinder_outer_radius_);
    if (-b_midpoint / cylinder_outer_radius_ <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(context,
                  "Bulge=true UniformCylindricalSide BRight outer constraint "
                  "-b_midpoint / CylinderOuterRadius > cos(72 deg) ~ 0.309 "
                  "violated: "
                      << -b_midpoint << " / " << cylinder_outer_radius_ << " = "
                      << -b_midpoint / cylinder_outer_radius_
                      << ". Try decreasing the star separation or "
                         "WedgeOuterRadius.");
    }
    const double a_midpoint =
        equal_arc_midpoint(0.0,
                           bulged_sphere_extent(rightmost_x_, d_aright, R_A,
                                                cylinder_outer_radius_),
                           cylinder_outer_radius_);
    if (a_midpoint / cylinder_outer_radius_ <= cos(std::numbers::pi * 0.4)) {
      PARSE_ERROR(context,
                  "Bulge=true UniformCylindricalSide ALeft outer constraint "
                  "a_midpoint / CylinderOuterRadius > cos(72 deg) ~ 0.309 "
                  "violated: "
                      << a_midpoint << " / " << cylinder_outer_radius_ << " = "
                      << a_midpoint / cylinder_outer_radius_
                      << ". Try decreasing the star separation or "
                         "WedgeOuterRadius.");
    }
  }

  set_shell_distribution(make_not_null(&number_of_spherical_shells_),
                         make_not_null(&spherical_shells_radial_distribution_),
                         spherical_shells_radial_partitioning_,
                         spherical_shells_radial_distribution,
                         cylinder_outer_radius_, outer_radius_,
                         "CylinderOuterRadius", "OuterRadius", context);

  // Expand the radial refinement into one entry per spherical shell. A single
  // value is used for all shells.
  if (std::holds_alternative<size_t>(spherical_shells_radial_refinement)) {
    spherical_shells_radial_refinement_ = std::vector<size_t>(
        number_of_spherical_shells_,
        std::get<size_t>(spherical_shells_radial_refinement));
  } else {
    spherical_shells_radial_refinement_ =
        std::get<std::vector<size_t>>(spherical_shells_radial_refinement);
    if (spherical_shells_radial_refinement_.size() !=
        number_of_spherical_shells_) {
      PARSE_ERROR(context,
                  "The number of radial refinement levels in "
                  "SphericalShellsInitialRadialRefinement ("
                      << spherical_shells_radial_refinement_.size()
                      << ") must match the number of spherical shells ("
                      << number_of_spherical_shells_
                      << "), which is one more than the number of entries in "
                         "SphericalShellsRadialPartitioning ("
                      << spherical_shells_radial_partitioning_.size() << ").");
    }
  }

  number_of_blocks_ = first_spherical_shell_block + number_of_spherical_shells_;

  // Create block names and groups
  block_names_.reserve(number_of_blocks_);
  auto add_cubed_cylinder_name = [this](const std::string& prefix) {
    for (const std::string& where :
         {"Center"s, "Front"s, "Top"s, "Back"s, "Bottom"s}) {
      const std::string name =
          std::string(prefix).append("CubedCylinder").append(where);
      block_names_.emplace_back(name);
      block_groups_["CubedCylinders"].insert(name);
      // The four non-central blocks of each cubed cylinder are the deformed
      // cube wedges surrounding the central cube.
      if (where != "Center"s) {
        block_groups_["Wedges"].insert(name);
      }
    }
  };

  // 4 cubed cylinders
  add_cubed_cylinder_name("BLeft");   // 0 - 4
  add_cubed_cylinder_name("BRight");  // 5 - 9
  add_cubed_cylinder_name("ALeft");   // 10-14
  add_cubed_cylinder_name("ARight");  // 15-19

  // 6 cylinders, between the deformed cubes and spherical shells
  block_names_.emplace_back("BFilledCylinder");
  block_groups_["Cylinders"].insert("BFilledCylinder");
  block_names_.emplace_back("BLeftHollowCylinder");
  block_groups_["Cylinders"].insert("BLeftHollowCylinder");
  block_names_.emplace_back("BRightHollowCylinder");
  block_groups_["Cylinders"].insert("BRightHollowCylinder");
  block_names_.emplace_back("ALeftHollowCylinder");
  block_groups_["Cylinders"].insert("ALeftHollowCylinder");
  block_names_.emplace_back("ARightHollowCylinder");
  block_groups_["Cylinders"].insert("ARightHollowCylinder");
  block_names_.emplace_back("AFilledCylinder");
  block_groups_["Cylinders"].insert("AFilledCylinder");

  // N spherical shells, one per radial partition
  for (size_t s = 0; s < number_of_spherical_shells_; ++s) {
    const std::string name = "SphericalShell" + std::to_string(s);
    block_names_.emplace_back(name);
    block_groups_["SphericalShells"].insert(name);
  }

  // initial grid points and refinement
  initial_grid_points_.resize(number_of_blocks_);
  initial_refinement_.resize(number_of_blocks_);

  // Blocks 0-19: 4 sets of cubed-cylinders (5 blocks each, stride 5).
  // Bulge=false: wedge prisms have xi piercing curved surface, eta "angular"
  //   along the curve, zeta along the extruded (axial) dimension.
  // Bulge=true: FOW blocks have xi along the axial dimension, eta "angular",
  //   zeta piercing the curved (sphere) surface.
  for (size_t g = 0; g < 4; ++g) {
    initial_grid_points_[5 * g] = make_array<3>(gsl::at(cube_grid_points_, g));
    initial_refinement_[5 * g] = {gsl::at(cube_x_refinement_, g),
                                  gsl::at(cube_yz_refinement_, g),
                                  gsl::at(cube_yz_refinement_, g)};
    for (size_t j = 1; j < 5; ++j) {
      if (bulge_) {
        // FOW: xi=axial, eta=angular, zeta=radial
        initial_grid_points_[5 * g + j] = {gsl::at(cube_grid_points_, g),
                                           gsl::at(cube_grid_points_, g),
                                           wedge_prism_radial_grid_points_};
        initial_refinement_[5 * g + j] = {gsl::at(cube_x_refinement_, g),
                                          gsl::at(cube_yz_refinement_, g),
                                          wedge_prism_radial_refinement_};
      } else {
        // 2D wedge "prism"
        initial_grid_points_[5 * g + j] = {wedge_prism_radial_grid_points_,
                                           gsl::at(cube_grid_points_, g),
                                           gsl::at(cube_grid_points_, g)};
        initial_refinement_[5 * g + j] = {wedge_prism_radial_refinement_,
                                          gsl::at(cube_yz_refinement_, g),
                                          gsl::at(cube_x_refinement_, g)};
      }
    }
  }
  // Blocks 20 & 25: filled cylinders (full_cylinder topology).
  // xi (B2, radial) is tied to (B2, angular); both cannot be h-refined. eta
  // must be odd. zeta (I1, axial) uses the cylinder radial parameters.
  // For both, zeta points along positive x, which is also true for the hollow
  // cylinders, so that the angular coordinates of all cylinders align (with
  // DiscreteRotations about x)
  if (b2_angular_grid_points_ % 2 == 0) {
    PARSE_ERROR(context, "B2InitialAngularGridPoints must be odd, but got "
                             << b2_angular_grid_points_ << ".");
  }
  const size_t theta_modes = b2_angular_grid_points_ / 2;
  // set by equating radial/angular modal space sizes for ZernikeB2
  const size_t b2_radial = theta_modes / 2 + 1 + theta_modes % 2;
  for (const size_t i : {b_filled_cylinder_block, a_filled_cylinder_block}) {
    initial_grid_points_[i] = {b2_radial, b2_angular_grid_points_,
                               cylinder_radial_grid_points_};
    initial_refinement_[i] = {0, 0, cylinder_radial_refinement_};
  }
  // Blocks 21 - 24: hollow cylinders (cylindrical_shell topology).
  // xi (I1, radial), eta (S1, angular) cannot be h-refined, must be odd.
  // zeta (I1, axial) uses the cylinder radial parameters.
  const auto [hollow_eta, hollow_zeta] = hollow_cylinder_angular_grid_points_;
  if (hollow_eta % 2 == 0) {
    PARSE_ERROR(context,
                "HollowCylinderInitialAngularGridPoints[0] must be odd, but "
                "got "
                    << hollow_eta << ".");
  }
  for (size_t i = first_hollow_cylinder_block; i < a_filled_cylinder_block;
       ++i) {
    initial_grid_points_[i] = {cylinder_radial_grid_points_, hollow_eta,
                               hollow_zeta};
    initial_refinement_[i] = {cylinder_radial_refinement_, 0, 0};
  }
  // Blocks 26 and up: spherical shells (spherical_shell topology).
  // xi (I1, radial)
  // eta (S2Colatitude) and zeta (S2Longitude) cannot be h-refined.
  // Spherepack requires zeta_pts = 2 * eta_pts - 1.
  const size_t colatitude_pts = spherical_harmonic_l_ + 1;
  for (size_t s = 0; s < number_of_spherical_shells_; ++s) {
    initial_grid_points_[first_spherical_shell_block + s] = {
        spherical_shells_radial_grid_points_, colatitude_pts,
        2 * colatitude_pts - 1};
    initial_refinement_[first_spherical_shell_block + s] = {
        spherical_shells_radial_refinement_[s], 0, 0};
  }

  grid_anchors_ =
      bco::create_grid_anchors({center_A_, 0., 0.}, {center_B_, 0., 0.});

  if (time_dependent_options_.has_value()) {
    if (time_dependent_options_->has_distorted_frame_options(
            domain::ObjectLabel::A) or
        time_dependent_options_->has_distorted_frame_options(
            domain::ObjectLabel::B)) {
      PARSE_ERROR(context,
                  "The BinaryNeutronStars domain does not support shape maps "
                  "because the objects are filled (no excision surfaces). Set "
                  "ShapeMapA and ShapeMapB to None.");
    }
    time_dependent_options_->build_maps(
        std::array{std::array<double, 3>{center_A_, 0., 0.},
                   std::array<double, 3>{center_B_, 0., 0.}},
        std::nullopt, std::nullopt, std::array<double, 3>{0., 0., 0.},
        std::optional<std::array<double, 2>>{std::nullopt},
        std::optional<std::array<double, 2>>{std::nullopt}, true, true,
        cylinder_outer_radius_, outer_radius_);
  }
}

Domain<3> BinaryNeutronStars::create_domain() const {
  using Interval = CoordinateMaps::Interval;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Wedge = CoordinateMaps::Wedge<2>;
  using WedgePrism = CoordinateMaps::ProductOf2Maps<Wedge, Affine>;
  using Identity1D = CoordinateMaps::Identity<1>;
  using Rotation = CoordinateMaps::DiscreteRotation<3>;
  using ShiftX = CoordinateMaps::ProductOf3Maps<Affine, Identity1D, Identity1D>;

  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::BlockLogical, Frame::Inertial, 3>>>
      coordinate_maps{};
  coordinate_maps.reserve(number_of_blocks_);
  // Checks that the most recently added map belongs to the block named `name`,
  // i.e. that the maps are added in the same order as block_names_.
  const auto check_block_name = [this,
                                 &coordinate_maps](const std::string& name) {
    const size_t block_id = coordinate_maps.size() - 1;
    if (block_names_[block_id] != name) {
      ERROR("Expected the map for block "
            << block_id << " to be for '" << block_names_[block_id]
            << "', but it is for '" << name << "'.");
    }
  };

  const auto aligned = OrientationMap<3>::create_aligned();
  const auto wedge_2d_aligned = OrientationMap<2>::create_aligned();

  // naming as starting from upper z axis
  const OrientationMap<3> rotate_to_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
      Direction<3>::lower_xi()}};

  const OrientationMap<3> rotate_to_minus_x_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_eta(),
      Direction<3>::upper_xi()}};

  const OrientationMap<3> half_turn_about_z(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});

  const OrientationMap<3> rotate_to_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
      Direction<3>::lower_eta()}};

  const OrientationMap<3> rotate_to_minus_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_zeta(),
      Direction<3>::upper_eta()}};

  const OrientationMap<3> rotate_to_minus_z_axis{std::array<Direction<3>, 3>{
      Direction<3>::upper_xi(), Direction<3>::lower_eta(),
      Direction<3>::lower_zeta()}};

  const OrientationMap<3> half_turn_about_y_minus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::lower_zeta(),
                                  Direction<3>::lower_eta()}};

  const OrientationMap<3> half_turn_about_y_plus_z_axis{
      std::array<Direction<3>, 3>{Direction<3>::lower_xi(),
                                  Direction<3>::upper_zeta(),
                                  Direction<3>::upper_eta()}};

  const OrientationMap<3> half_turn_about_y_axis{std::array<Direction<3>, 3>{
      Direction<3>::lower_xi(), Direction<3>::upper_eta(),
      Direction<3>::lower_zeta()}};

  // For Bulge=false: Wedges must first apply rotate_to_minus_x_axis, then
  // rotate about x axis
  const OrientationMap<3> rotate_wedge_front{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::lower_xi(),
      Direction<3>::upper_eta()}};

  const OrientationMap<3> rotate_wedge_top = rotate_to_minus_x_axis;

  const OrientationMap<3> rotate_wedge_back{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::upper_xi(),
      Direction<3>::lower_eta()}};

  const OrientationMap<3> rotate_wedge_bottom{std::array<Direction<3>, 3>{
      Direction<3>::lower_zeta(), Direction<3>::lower_eta(),
      Direction<3>::lower_xi()}};

  // For Bulge=true: radii of the spheres around each object (computed in the
  // constructor) so that both FOW groups meet the x = 0 plane in the same
  // circle
  const double R_B = bulge_radius_B_;
  const double R_A = bulge_radius_A_;
  // Half-width of the central cubes in y and z, which is also the FOW
  // flat-face half-width
  const double L = wedge_inner_radius_ / std::numbers::sqrt2;

  // Blocks 0 - 19
  if (not bulge_) {
    // Wedges are extruded 2D wedges
    auto add_cubed_cylinder = [this, &coordinate_maps, &check_block_name,
                               &rotate_wedge_front, &rotate_wedge_top,
                               &rotate_wedge_back, &rotate_wedge_bottom,
                               &wedge_2d_aligned,
                               L](const double lower_xi, const double upper_xi,
                                  const std::string& group) {
      // central cube
      coordinate_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{Affine{-1.0, 1.0, lower_xi, upper_xi},
                       Affine{-1.0, 1.0, -L, L}, Affine{-1.0, 1.0, -L, L}}));
      check_block_name(group + "CubedCylinderCenter"s);
      // 4 surrounding wedges (Front, Top, Back, Bottom)
      const std::array<const char*, 4> wedge_wheres{"Front", "Top", "Back",
                                                    "Bottom"};
      size_t where_idx = 0;
      for (const auto& rotation : {rotate_wedge_front, rotate_wedge_top,
                                   rotate_wedge_back, rotate_wedge_bottom}) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                WedgePrism{Wedge{wedge_inner_radius_, wedge_outer_radius_, 0.0,
                                 1.0, wedge_2d_aligned, false},
                           Affine{-1.0, 1.0, -upper_xi, -lower_xi}},
                Rotation{rotation}));
        check_block_name(group + "CubedCylinder"s +
                         gsl::at(wedge_wheres, where_idx++));
      }
    };
    add_cubed_cylinder(leftmost_x_, center_B_, "BLeft");
    add_cubed_cylinder(center_B_, 0.0, "BRight");
    add_cubed_cylinder(0.0, center_A_, "ALeft");
    add_cubed_cylinder(center_A_, rightmost_x_, "ARight");
  } else {
    // Bulge=true: FOW blocks
    const auto add_central_cube_block = [&coordinate_maps, &check_block_name,
                                         L](const double lower_x,
                                            const double upper_x,
                                            const std::string& group) {
      coordinate_maps.emplace_back(
          make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
              Affine3D{Affine{-1.0, 1.0, lower_x, upper_x},
                       Affine{-1.0, 1.0, -L, L}, Affine{-1.0, 1.0, -L, L}}));
      check_block_name(group + "CubedCylinderCenter"s);
    };

    // BLeft (x in [leftmost_x_, center_B_])
    add_central_cube_block(leftmost_x_, center_B_, "BLeft");
    {
      const double d_bl = center_B_ - leftmost_x_;
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_bl{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_bl) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, d_bl, R_B},
                ShiftX{Affine{0.0, d_bl, leftmost_x_, center_B_}, Identity1D{},
                       Identity1D{}},
                Rotation{map}));
        check_block_name("BLeftCubedCylinder"s + where);
      }
    }

    // BRight (x in [center_B_, 0])
    add_central_cube_block(center_B_, 0.0, "BRight");
    {
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_br{{{half_turn_about_y_minus_z_axis, "Front"},
                       {half_turn_about_z, "Top"},
                       {half_turn_about_y_plus_z_axis, "Back"},
                       {half_turn_about_y_axis, "Bottom"}}};
      for (const auto& [combined, where] : fow_rots_br) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, std::abs(center_B_), R_B},
                Rotation{combined}));
        check_block_name("BRightCubedCylinder"s + where);
      }
    }

    // ALeft (x in [0, center_A_])
    add_central_cube_block(0.0, center_A_, "ALeft");
    {
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_al{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_al) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, center_A_, R_A},
                ShiftX{Affine{0.0, center_A_, 0.0, center_A_}, Identity1D{},
                       Identity1D{}},
                Rotation{map}));
        check_block_name("ALeftCubedCylinder"s + where);
      }
    }

    // ARight (x in [center_A_, rightmost_x_])
    add_central_cube_block(center_A_, rightmost_x_, "ARight");
    {
      const double d_ar = rightmost_x_ - center_A_;
      const std::array<std::pair<OrientationMap<3>, const char*>, 4>
          fow_rots_ar{{{rotate_to_minus_y_axis, "Front"},
                       {aligned, "Top"},
                       {rotate_to_y_axis, "Back"},
                       {rotate_to_minus_z_axis, "Bottom"}}};
      for (const auto& [map, where] : fow_rots_ar) {
        coordinate_maps.emplace_back(
            make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
                CoordinateMaps::FlatOffsetWedge{L, d_ar, R_A},
                Rotation{half_turn_about_z},
                ShiftX{Affine{-d_ar, 0.0, center_A_, rightmost_x_},
                       Identity1D{}, Identity1D{}},
                Rotation{map}));
        check_block_name("ARightCubedCylinder"s + where);
      }
    }
  }

  // Lambdas for cylinder and sphere-shell blocks

  // ZernikeB2 filled-cylinder (full_cylinder topology).
  const auto add_filled_cylinder =
      [&coordinate_maps, &check_block_name, &rotate_to_minus_x_axis,
       &half_turn_about_y_axis, &aligned](
          const CoordinateMaps::CylindricalFlatEndcapInterior& endcap_map,
          const Rotation& rotation_map, const std::string& name) {
        const bool minus_side =
            rotation_map == Rotation{rotate_to_minus_x_axis};
        coordinate_maps.push_back(
            domain::make_coordinate_map_base<Frame::BlockLogical,
                                             Frame::Inertial>(
                ShiftX{Affine{-1.0, 1.0, 0.0, 1.0}, Identity1D{}, Identity1D{}},
                CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                               Identity1D>{
                    CoordinateMaps::PolarToCartesian{}, Identity1D{}},
                Rotation{minus_side ? half_turn_about_y_axis : aligned},
                endcap_map, rotation_map));
        check_block_name(name);
      };

  // Bulge=false hollow cylinder: CylindricalSphericalShell.
  const auto add_hollow_cylinder_flat = [this, &coordinate_maps,
                                         &check_block_name, &rotate_to_y_axis](
                                            const double x_inner_lower,
                                            const double x_inner_upper,
                                            const double x_outer_lower,
                                            const double x_outer_upper,
                                            const std::string& name) {
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::CylindricalSphericalShell{
                x_inner_lower, x_inner_upper, x_outer_lower, x_outer_upper,
                wedge_outer_radius_, cylinder_outer_radius_},
            Rotation{rotate_to_y_axis}));
    check_block_name(name);
  };

  // Bulge=true hollow cylinder: UniformCylindricalSide.
  const auto add_hollow_cylinder_bulged = [this, &coordinate_maps,
                                           &check_block_name,
                                           &rotate_to_minus_x_axis,
                                           &half_turn_about_y_axis, &aligned](
                                              const double center,
                                              const double inner_radius,
                                              const double z_inner_upper,
                                              const double z_inner_lower,
                                              const double z_outer_upper,
                                              const double z_outer_lower,
                                              const Rotation& rotation_map,
                                              const std::string& name) {
    const bool minus_side = rotation_map == Rotation{rotate_to_minus_x_axis};
    coordinate_maps.push_back(
        domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
            CoordinateMaps::ProductOf3Maps<Affine, Identity1D,
                                           CoordinateMaps::Interval>{
                Affine{-1.0, 1.0, 1.0, 2.0}, Identity1D{},
                CoordinateMaps::Interval{-1.0, 1.0, -1.0, 1.0,
                                         CoordinateMaps::Distribution::Linear}},
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::PolarToCartesian,
                                           Identity1D>{
                CoordinateMaps::PolarToCartesian{}, Identity1D{}},
            Rotation{minus_side ? half_turn_about_y_axis : aligned},
            CoordinateMaps::UniformCylindricalSide(
                std::array<double, 3>{0.0, 0.0, center}, make_array<3>(0.0),
                inner_radius, cylinder_outer_radius_, z_inner_upper,
                z_inner_lower, z_outer_upper, z_outer_lower),
            rotation_map));
    check_block_name(name);
  };

  // Filled cylinders, hollow cylinders, and spherical shells.
  // filled_edge_x_b and filled_edge_x_a are the x-coordinates on the sphere of
  // radius CylinderOuterRadius where the filled cylinders meet the hollow
  // cylinders. They are the radial projections from the origin of the circles
  // where the planes x = LeftEnd and x = RightEnd cut the outer faces of the
  // deformed-cube wedges, so they depend on Bulge.
  const double filled_edge_x_b =
      bulge_ ? bulged_sphere_extent(leftmost_x_, center_B_ - leftmost_x_, R_B,
                                    cylinder_outer_radius_)
             : cylinder_outer_radius_ * leftmost_x_ /
                   sqrt(square(wedge_outer_radius_) + square(leftmost_x_));
  const double filled_edge_x_a =
      bulge_ ? bulged_sphere_extent(rightmost_x_, rightmost_x_ - center_A_, R_A,
                                    cylinder_outer_radius_)
             : cylinder_outer_radius_ * rightmost_x_ /
                   sqrt(square(wedge_outer_radius_) + square(rightmost_x_));

  const auto get_x_midpoint = [this](const double lower, const double upper) {
    return equal_arc_midpoint(lower, upper, cylinder_outer_radius_);
  };

  // Block 20: BFilledCylinder
  add_filled_cylinder(CoordinateMaps::CylindricalFlatEndcapInterior(
                          std::array<double, 3>{0.0, 0.0, leftmost_x_},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          std::array<double, 3>{0.0, 0.0, 0.0}, filled_edge_x_b,
                          cylinder_outer_radius_),
                      Rotation{rotate_to_x_axis}, "BFilledCylinder");

  // Blocks 21-24: hollow cylinders.
  if (not bulge_) {
    const double central_midpoint =
        get_x_midpoint(filled_edge_x_b, filled_edge_x_a);
    const double b_midpoint = get_x_midpoint(filled_edge_x_b, central_midpoint);
    const double a_midpoint = get_x_midpoint(central_midpoint, filled_edge_x_a);

    add_hollow_cylinder_flat(leftmost_x_, center_B_, filled_edge_x_b,
                             b_midpoint, "BLeftHollowCylinder");
    add_hollow_cylinder_flat(center_B_, 0.0, b_midpoint, central_midpoint,
                             "BRightHollowCylinder");
    add_hollow_cylinder_flat(0.0, center_A_, central_midpoint, a_midpoint,
                             "ALeftHollowCylinder");
    add_hollow_cylinder_flat(center_A_, rightmost_x_, a_midpoint,
                             filled_edge_x_a, "ARightHollowCylinder");
  } else {
    // Having central_midpoint be shifted causes issues with the
    // UniformCylindricalSide map
    const double central_midpoint = 0.0;
    const double b_midpoint = get_x_midpoint(filled_edge_x_b, central_midpoint);
    const double a_midpoint = get_x_midpoint(central_midpoint, filled_edge_x_a);
    // The UniformCylindricalSide outer constraints on b_midpoint and
    // a_midpoint are checked in the constructor.
    add_hollow_cylinder_bulged(
        -center_B_, R_B, -leftmost_x_, -center_B_, -filled_edge_x_b,
        -b_midpoint, Rotation{rotate_to_minus_x_axis}, "BLeftHollowCylinder");
    add_hollow_cylinder_bulged(
        -center_B_, R_B, -center_B_, 0.0, -b_midpoint, central_midpoint,
        Rotation{rotate_to_minus_x_axis}, "BRightHollowCylinder");
    add_hollow_cylinder_bulged(center_A_, R_A, center_A_, 0.0, a_midpoint,
                               central_midpoint, Rotation{rotate_to_x_axis},
                               "ALeftHollowCylinder");
    add_hollow_cylinder_bulged(
        center_A_, R_A, rightmost_x_, center_A_, filled_edge_x_a, a_midpoint,
        Rotation{rotate_to_x_axis}, "ARightHollowCylinder");
  }

  // Block 25: AFilledCylinder
  add_filled_cylinder(CoordinateMaps::CylindricalFlatEndcapInterior(
                          std::array<double, 3>{0.0, 0.0, -rightmost_x_},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          std::array<double, 3>{0.0, 0.0, 0.0},
                          -filled_edge_x_a, cylinder_outer_radius_),
                      Rotation{rotate_to_minus_x_axis}, "AFilledCylinder");

  // Blocks 26 and up: SphericalShell0, SphericalShell1, ...
  // S2 spherical-harmonic shells, one per radial partition between
  // cylinder_outer_radius_ and outer_radius_.
  {
    double inner_radius = cylinder_outer_radius_;
    for (size_t s = 0; s < number_of_spherical_shells_; ++s) {
      const double outer_radius = (s + 1 < number_of_spherical_shells_)
                                      ? spherical_shells_radial_partitioning_[s]
                                      : outer_radius_;
      coordinate_maps.push_back(domain::make_coordinate_map_base<
                                Frame::BlockLogical, Frame::Inertial>(
          CoordinateMaps::ProductOf2Maps<Interval, CoordinateMaps::Identity<2>>{
              Interval{-1.0, 1.0, inner_radius, outer_radius,
                       spherical_shells_radial_distribution_[s], 0.0},
              CoordinateMaps::Identity<2>{}},
          CoordinateMaps::SphericalToCartesianPfaffian{}));
      check_block_name("SphericalShell" + std::to_string(s));
      inner_radius = outer_radius;
    }
  }

  // Neighbor graph
  using Dir = Direction<3>;

  // Directed neighbor graph keyed by block name:
  // {self name, self direction, neighbor name, orientation of neighbor}.
  struct EdgeSpec {
    std::string self_name;
    Dir self_direction;
    std::string neighbor_name;
    OrientationMap<3> orientation;
    bool are_conforming = true;
  };
  std::vector<EdgeSpec> edges;

  std::unordered_map<std::string, size_t> block_index{};
  for (size_t i = 0; i < block_names_.size(); ++i) {
    block_index.emplace(block_names_[i], i);
  }

  // Central cubes are always conforming and use aligned orientation.
  // Note: each element is spelled out with its type because nvcc rejects the
  // brace-elided form `{std::pair{...}, {...}, {...}}`.
  using TagPair = std::pair<std::string, std::string>;
  for (const auto& [a, b] :
       std::array{TagPair{"BLeft"s, "BRight"s}, TagPair{"BRight"s, "ALeft"s},
                  TagPair{"ALeft"s, "ARight"s}}) {
    edges.push_back({a + "CubedCylinderCenter", Dir::upper_xi(),
                     b + "CubedCylinderCenter", aligned});
    edges.push_back({b + "CubedCylinderCenter", Dir::lower_xi(),
                     a + "CubedCylinderCenter", aligned});
  }

  if (not bulge_) {
    // Bulge=false: WedgePrism blocks connect to central cubes via
    // rotate_wedge_* and to each other at axial (zeta) seams with aligned
    // orientation.
    const std::vector<
        std::tuple<std::string, Dir, std::string, OrientationMap<3>>>
        cube_face_defs{
            {"BLeft", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"BLeft", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"BLeft", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"BLeft", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"ARight", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"ARight", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"ARight", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"ARight", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"ALeft", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"ALeft", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"ALeft", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"ALeft", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
            {"BRight", Dir::lower_eta(), "Front", rotate_wedge_front},
            {"BRight", Dir::upper_zeta(), "Top", rotate_wedge_top},
            {"BRight", Dir::upper_eta(), "Back", rotate_wedge_back},
            {"BRight", Dir::lower_zeta(), "Bottom", rotate_wedge_bottom},
        };
    for (const auto& [grp, face_dir, where, rot] : cube_face_defs) {
      const std::string cube = std::string(grp).append("CubedCylinderCenter"s);
      const std::string wedge =
          std::string(grp).append("CubedCylinder"s).append(where);
      edges.push_back({cube, face_dir, wedge, rot});
      edges.push_back({wedge, Dir::lower_xi(), cube, rot.inverse_map()});
    }

    // Axial seams between groups (zeta direction, aligned).
    for (const std::string where : {"Front", "Top", "Back", "Bottom"}) {
      // x = center_B_
      edges.push_back({"BLeftCubedCylinder" + where, Dir::lower_zeta(),
                       "BRightCubedCylinder" + where, aligned});
      edges.push_back({"BRightCubedCylinder" + where, Dir::upper_zeta(),
                       "BLeftCubedCylinder" + where, aligned});
      // x = 0
      edges.push_back({"BRightCubedCylinder" + where, Dir::lower_zeta(),
                       "ALeftCubedCylinder" + where, aligned});
      edges.push_back({"ALeftCubedCylinder" + where, Dir::upper_zeta(),
                       "BRightCubedCylinder" + where, aligned});
      // x = center_A_
      edges.push_back({"ALeftCubedCylinder" + where, Dir::lower_zeta(),
                       "ARightCubedCylinder" + where, aligned});
      edges.push_back({"ARightCubedCylinder" + where, Dir::upper_zeta(),
                       "ALeftCubedCylinder" + where, aligned});
    }
  } else {
    // Bulge=true
    const std::vector<
        std::tuple<std::string, Dir, std::string, OrientationMap<3>>>
        fow_cube_defs{
            // {group, cube_face_dir, where, map}
            {"BLeft", Dir::lower_eta(), "Front", rotate_to_minus_y_axis},
            {"BLeft", Dir::upper_zeta(), "Top", aligned},
            {"BLeft", Dir::upper_eta(), "Back", rotate_to_y_axis},
            {"BLeft", Dir::lower_zeta(), "Bottom", rotate_to_minus_z_axis},
            {"ALeft", Dir::lower_eta(), "Front", rotate_to_minus_y_axis},
            {"ALeft", Dir::upper_zeta(), "Top", aligned},
            {"ALeft", Dir::upper_eta(), "Back", rotate_to_y_axis},
            {"ALeft", Dir::lower_zeta(), "Bottom", rotate_to_minus_z_axis},
            // BRight
            {"BRight", Dir::lower_eta(), "Front",
             half_turn_about_y_minus_z_axis},
            {"BRight", Dir::upper_zeta(), "Top", half_turn_about_z},
            {"BRight", Dir::upper_eta(), "Back", half_turn_about_y_plus_z_axis},
            {"BRight", Dir::lower_zeta(), "Bottom", half_turn_about_y_axis},
            // ARight
            {"ARight", Dir::lower_eta(), "Front",
             half_turn_about_y_minus_z_axis},
            {"ARight", Dir::upper_zeta(), "Top", half_turn_about_z},
            {"ARight", Dir::upper_eta(), "Back", half_turn_about_y_plus_z_axis},
            {"ARight", Dir::lower_zeta(), "Bottom", half_turn_about_y_axis}};
    for (const auto& [grp, face_dir, where, map] : fow_cube_defs) {
      const std::string cube = std::string(grp).append("CubedCylinderCenter"s);
      const std::string fow =
          std::string(grp).append("CubedCylinder"s).append(where);
      edges.push_back({cube, face_dir, fow, map});
      edges.push_back({fow, Dir::lower_zeta(), cube, map.inverse_map()});
    }

    // Axial seams between FOW groups (xi direction)
    for (const std::string where : {"Front", "Top", "Back", "Bottom"}) {
      edges.push_back({"BLeftCubedCylinder" + where, Dir::upper_xi(),
                       "BRightCubedCylinder" + where, half_turn_about_z});
      edges.push_back({"BRightCubedCylinder" + where, Dir::upper_xi(),
                       "BLeftCubedCylinder" + where, half_turn_about_z});
      edges.push_back({"BRightCubedCylinder" + where, Dir::lower_xi(),
                       "ALeftCubedCylinder" + where, half_turn_about_z});
      edges.push_back({"ALeftCubedCylinder" + where, Dir::lower_xi(),
                       "BRightCubedCylinder" + where, half_turn_about_z});
      edges.push_back({"ALeftCubedCylinder" + where, Dir::upper_xi(),
                       "ARightCubedCylinder" + where, half_turn_about_z});
      edges.push_back({"ARightCubedCylinder" + where, Dir::upper_xi(),
                       "ALeftCubedCylinder" + where, half_turn_about_z});
    }
  }

  // Angular connections within each group (aligned orientation in all cases)
  const std::array<std::string, 4> sides{"Front", "Top", "Back", "Bottom"};
  for (const std::string grp : {"BLeft", "BRight", "ALeft", "ARight"}) {
    const bool reversed = bulge_ and (grp == "BRight" or grp == "ARight");
    for (size_t i = 0; i < 4; ++i) {
      const std::string a = grp + "CubedCylinder" + gsl::at(sides, i);
      const std::string b = grp + "CubedCylinder" + gsl::at(sides, (i + 1) % 4);
      if (reversed) {
        edges.push_back({a, Dir::lower_eta(), b, aligned});
        edges.push_back({b, Dir::upper_eta(), a, aligned});
      } else {
        edges.push_back({a, Dir::upper_eta(), b, aligned});
        edges.push_back({b, Dir::lower_eta(), a, aligned});
      }
    }
  }

  // Cylinder connections
  const OrientationMap<3> orient_b_filled_to_hollow{
      std::array<Dir, 3>{Dir::upper_zeta(), Dir::upper_eta(), Dir::lower_xi()}};

  const OrientationMap<3> orient_a_filled_to_hollow{
      std::array<Dir, 3>{Dir::lower_zeta(), Dir::upper_eta(), Dir::upper_xi()}};

  edges.push_back({"BFilledCylinder", Dir::upper_xi(), "BLeftHollowCylinder",
                   orient_b_filled_to_hollow});
  edges.push_back({"BLeftHollowCylinder", Dir::lower_zeta(), "BFilledCylinder",
                   orient_b_filled_to_hollow.inverse_map()});
  edges.push_back({"AFilledCylinder", Dir::upper_xi(), "ARightHollowCylinder",
                   orient_a_filled_to_hollow});
  edges.push_back({"ARightHollowCylinder", Dir::upper_zeta(), "AFilledCylinder",
                   orient_a_filled_to_hollow.inverse_map()});

  edges.push_back({"BLeftHollowCylinder", Dir::upper_zeta(),
                   "BRightHollowCylinder", aligned});
  edges.push_back({"BRightHollowCylinder", Dir::lower_zeta(),
                   "BLeftHollowCylinder", aligned});
  edges.push_back({"BRightHollowCylinder", Dir::upper_zeta(),
                   "ALeftHollowCylinder", aligned});
  edges.push_back({"ALeftHollowCylinder", Dir::lower_zeta(),
                   "BRightHollowCylinder", aligned});
  edges.push_back({"ALeftHollowCylinder", Dir::upper_zeta(),
                   "ARightHollowCylinder", aligned});
  edges.push_back({"ARightHollowCylinder", Dir::lower_zeta(),
                   "ALeftHollowCylinder", aligned});

  // Conforming radial seams between adjacent spherical shells.
  for (size_t s = 0; s + 1 < number_of_spherical_shells_; ++s) {
    const std::string inner = "SphericalShell" + std::to_string(s);
    const std::string outer = "SphericalShell" + std::to_string(s + 1);
    edges.push_back({inner, Dir::upper_xi(), outer, aligned});
    edges.push_back({outer, Dir::lower_xi(), inner, aligned});
  }

  std::vector<DirectionMap<3, BlockNeighbors<3>>> neighbors(
      coordinate_maps.size());
  for (const auto& edge : edges) {
    const auto self_it = block_index.find(edge.self_name);
    const auto neighbor_it = block_index.find(edge.neighbor_name);
    if (self_it == block_index.end() or neighbor_it == block_index.end()) {
      ERROR("Unknown block name in the neighbor graph: '"
            << edge.self_name << "' or '" << edge.neighbor_name << "'.");
    }
    neighbors[self_it->second].emplace(
        edge.self_direction,
        BlockNeighbors<3>({neighbor_it->second},
                          {{neighbor_it->second, edge.orientation}},
                          edge.are_conforming));
  }

  // Non-conforming interface between the innermost spherical shell and the
  // outer cylinders.
  {
    const OrientationMap<3> sphere_to_b_filled{
        std::array<Dir, 3>{Dir::lower_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_a_filled{
        std::array<Dir, 3>{Dir::upper_zeta(), Dir::self(), Dir::self()}};
    const OrientationMap<3> sphere_to_side{
        std::array<Dir, 3>{Dir::upper_xi(), Dir::self(), Dir::self()}};
    const size_t c = block_index.at("SphericalShell0");
    const size_t bf = block_index.at("BFilledCylinder");
    const size_t blh = block_index.at("BLeftHollowCylinder");
    const size_t brh = block_index.at("BRightHollowCylinder");
    const size_t alh = block_index.at("ALeftHollowCylinder");
    const size_t arh = block_index.at("ARightHollowCylinder");
    const size_t af = block_index.at("AFilledCylinder");
    neighbors[c].emplace(Dir::lower_xi(),
                         BlockNeighbors<3>{{bf, blh, brh, alh, arh, af},
                                           {{bf, sphere_to_b_filled},
                                            {blh, sphere_to_side},
                                            {brh, sphere_to_side},
                                            {alh, sphere_to_side},
                                            {arh, sphere_to_side},
                                            {af, sphere_to_a_filled}},
                                           false});
    neighbors[bf].emplace(
        Dir::lower_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_b_filled.inverse_map()}}, false});
    neighbors[af].emplace(
        Dir::upper_zeta(),
        BlockNeighbors<3>{{c}, {{c, sphere_to_a_filled.inverse_map()}}, false});
    for (const size_t hc : {blh, brh, alh, arh}) {
      neighbors[hc].emplace(
          Dir::upper_xi(),
          BlockNeighbors<3>{{c}, {{c, sphere_to_side.inverse_map()}}, false});
    }
  }

  // Non-conforming interfaces between deformed cubes and hollow cylinders.
  {
    const OrientationMap<3> cube_to_hollow =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::upper_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_xi(), Dir::self(), Dir::self()}};
    const OrientationMap<3> hollow_to_cube =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{Dir::upper_zeta(),
                                                      Dir::self(), Dir::self()}}
               : OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_xi(), Dir::self(), Dir::self()}};
    const Dir cube_outer_face = bulge_ ? Dir::upper_zeta() : Dir::upper_xi();

    for (const std::string group : {"BLeft", "BRight", "ALeft", "ARight"}) {
      const size_t hollow = block_index.at(group + "HollowCylinder");
      std::unordered_set<size_t> wedges{};
      std::unordered_map<size_t, OrientationMap<3>> orientations{};
      for (const std::string where : {"Front", "Top", "Back", "Bottom"}) {
        const size_t wedge = block_index.at(group + "CubedCylinder" + where);
        wedges.insert(wedge);
        orientations.emplace(wedge, hollow_to_cube);
        neighbors[wedge].emplace(
            cube_outer_face,
            BlockNeighbors<3>{{hollow}, {{hollow, cube_to_hollow}}, false});
      }
      neighbors[hollow].emplace(
          Dir::lower_xi(),
          BlockNeighbors<3>{std::move(wedges), std::move(orientations), false});
    }
  }

  {
    // BFilledCylinder <-> BLeft
    const OrientationMap<3> bfc_to_blcc{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_xi()}};
    const OrientationMap<3> bfc_to_bldeformed =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_zeta(), Dir::self(), Dir::upper_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::lower_zeta()}};
    const Dir bl_fc_face = bulge_ ? Dir::lower_xi() : Dir::upper_zeta();

    const size_t bf = block_index.at("BFilledCylinder");
    const size_t blcc = block_index.at("BLeftCubedCylinderCenter");
    const size_t blf = block_index.at("BLeftCubedCylinderFront");
    const size_t blt = block_index.at("BLeftCubedCylinderTop");
    const size_t blb = block_index.at("BLeftCubedCylinderBack");
    const size_t blbo = block_index.at("BLeftCubedCylinderBottom");
    neighbors[bf].emplace(Dir::upper_zeta(),
                          BlockNeighbors<3>{{blcc, blf, blt, blb, blbo},
                                            {{blcc, bfc_to_blcc},
                                             {blf, bfc_to_bldeformed},
                                             {blt, bfc_to_bldeformed},
                                             {blb, bfc_to_bldeformed},
                                             {blbo, bfc_to_bldeformed}},
                                            false});
    neighbors[blcc].emplace(
        Dir::lower_xi(),
        BlockNeighbors<3>{{bf}, {{bf, bfc_to_blcc.inverse_map()}}, false});
    for (const size_t idx : {blf, blt, blb, blbo}) {
      neighbors[idx].emplace(
          bl_fc_face,
          BlockNeighbors<3>{
              {bf}, {{bf, bfc_to_bldeformed.inverse_map()}}, false});
    }

    // AFilledCylinder <-> ARight
    const OrientationMap<3> afc_to_arcc{
        std::array<Dir, 3>{Dir::self(), Dir::self(), Dir::upper_xi()}};
    const OrientationMap<3> afc_to_ardeformed =
        bulge_ ? OrientationMap<3>{std::array<Dir, 3>{
                     Dir::upper_zeta(), Dir::self(), Dir::lower_xi()}}
               : OrientationMap<3>{std::array<Dir, 3>{Dir::self(), Dir::self(),
                                                      Dir::lower_zeta()}};
    const Dir ar_fc_face = bulge_ ? Dir::lower_xi() : Dir::lower_zeta();

    const size_t af = block_index.at("AFilledCylinder");
    const size_t arcc = block_index.at("ARightCubedCylinderCenter");
    const size_t arf = block_index.at("ARightCubedCylinderFront");
    const size_t art = block_index.at("ARightCubedCylinderTop");
    const size_t arb = block_index.at("ARightCubedCylinderBack");
    const size_t arbo = block_index.at("ARightCubedCylinderBottom");
    neighbors[af].emplace(Dir::lower_zeta(),
                          BlockNeighbors<3>{{arcc, arf, art, arb, arbo},
                                            {{arcc, afc_to_arcc},
                                             {arf, afc_to_ardeformed},
                                             {art, afc_to_ardeformed},
                                             {arb, afc_to_ardeformed},
                                             {arbo, afc_to_ardeformed}},
                                            false});
    neighbors[arcc].emplace(
        Dir::upper_xi(),
        BlockNeighbors<3>{{af}, {{af, afc_to_arcc.inverse_map()}}, false});
    for (const size_t idx : {arf, art, arb, arbo}) {
      neighbors[idx].emplace(
          ar_fc_face,
          BlockNeighbors<3>{
              {af}, {{af, afc_to_ardeformed.inverse_map()}}, false});
    }
  }

  std::unordered_map<std::string, ExcisionSphere<3>> excision_spheres{};

  std::vector<Block<3>> blocks{};
  blocks.reserve(number_of_blocks_);
  for (size_t i = 0; i < coordinate_maps.size(); ++i) {
    std::array<domain::Topology, 3> topology = domain::topologies::hypercube<3>;
    if (i >= first_spherical_shell_block) {
      topology = domain::topologies::spherical_shell;
    } else if (i == b_filled_cylinder_block or i == a_filled_cylinder_block) {
      topology = domain::topologies::full_cylinder;
    } else if (i >= first_hollow_cylinder_block) {
      topology = domain::topologies::cylindrical_shell;
    }
    blocks.emplace_back(std::move(coordinate_maps[i]), i,
                        std::move(neighbors[i]), block_names_[i], topology);
  }

  Domain<3> domain{std::move(blocks), std::move(excision_spheres),
                   block_groups_};

  if (time_dependent_options_.has_value()) {
    // All blocks get only a Grid->Inertial map (no distorted frame since there
    // are no excision surfaces / shape maps in the BinaryNeutronStars domain).
    // Blocks 0-25 (inner region): rigid RotScaleTrans.
    // Blocks 26 and up (spherical shells): transitioning RotScaleTrans.
    std::vector<std::unique_ptr<
        domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>>>
        grid_to_inertial_block_maps{number_of_blocks_};
    for (size_t block_id = 0; block_id < first_spherical_shell_block;
         ++block_id) {
      grid_to_inertial_block_maps[block_id] =
          time_dependent_options_
              ->grid_to_inertial_map<domain::ObjectLabel::None>(false, true);
    }
    for (size_t block_id = first_spherical_shell_block;
         block_id < number_of_blocks_; ++block_id) {
      grid_to_inertial_block_maps[block_id] =
          time_dependent_options_
              ->grid_to_inertial_map<domain::ObjectLabel::None>(false, false);
    }
    for (size_t block_id = 0; block_id < number_of_blocks_; ++block_id) {
      if (grid_to_inertial_block_maps[block_id] == nullptr) {
        continue;
      }
      domain.inject_time_dependent_map_for_block(
          block_id, std::move(grid_to_inertial_block_maps[block_id]), nullptr,
          nullptr);
    }
  }
  return domain;
}

std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
BinaryNeutronStars::external_boundary_conditions() const {
  if (outer_boundary_condition_ == nullptr) {
    return {};
  }
  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
      boundary_conditions{number_of_blocks_};
  // Outer boundary: upper_xi face of the outermost spherical shell.
  boundary_conditions[number_of_blocks_ - 1][Direction<3>::upper_xi()] =
      outer_boundary_condition_->get_clone();
  return boundary_conditions;
}

std::vector<std::array<size_t, 3>> BinaryNeutronStars::initial_extents() const {
  return initial_grid_points_;
}

std::vector<std::array<size_t, 3>>
BinaryNeutronStars::initial_refinement_levels() const {
  return initial_refinement_;
}

std::unordered_map<std::string,
                   std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
BinaryNeutronStars::functions_of_time(
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
