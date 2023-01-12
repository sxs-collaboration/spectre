// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Interval;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t VolumeDim>
class Wedge;
template <size_t VolumeDim>
class DiscreteRotation;
class UniformCylindricalEndcap;
class UniformCylindricalFlatEndcap;
class UniformCylindricalSide;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;

namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain

namespace Frame {
struct Inertial;
struct BlockLogical;
}  // namespace Frame
/// \endcond

namespace domain::creators {

/*!
 * \ingroup ComputationalDomainGroup
 *
 * \brief A general domain for two compact objects based on cylinders.
 *
 * Creates a 3D Domain that represents a binary compact object
 * solution.  This domain is described briefly in the Appendix of
 * \cite Buchman:2012dw, and is illustrated in Figure 20 of that
 * paper.
 *
 * In the code and options below, `ObjectA` and `ObjectB` refer to the
 * two compact objects. In the grid frame, `ObjectA` is located to the
 * right of (i.e. a more positive value of the x-coordinate than)
 * `ObjectB`.  The inner edge of the Blocks surrounding each of
 * `ObjectA` and `ObjectB` is spherical in grid coordinates; the
 * user must specify the center and radius of this surface for both
 * `ObjectA` and `ObjectB`, and the user must specify the outer boundary
 * radius.  The outer boundary is a sphere centered at the origin.
 *
 * Note that Figure 20 of \cite Buchman:2012dw illustrates additional
 * spherical shells inside the "EA" and "EB" blocks, and the caption
 * of Figure 20 indicates that there are additional spherical shells
 * outside the "CA" and "CB" blocks; `CylindricalBinaryCompactObject`
 * has these extra shells inside "EA" only if the option `IncludeInnerSphereA`
 * is true, it has the extra shells inside "EB" only if the option
 * `IncludeInnerSphereB` is true, and it has the extra shells outside
 * "CA" and "CB" only if `IncludeOuterSphere` is true.
 * If the shells are absent, then the "EA" and "EB"
 * blocks extend to the excision boundaries and the "CA" and "CB" blocks
 * extend to the outer boundary.
 *
 * The Blocks are named as follows:
 * - Each of CAFilledCylinder, EAFilledCylinder, EBFilledCylinder,
 *   MAFilledCylinder, MBFilledCylinder, and CBFilledCylinder consists
 *   of 5 blocks, named 'Center', 'East', 'North', 'West', and
 *   'South', so an example of a valid block name is
 *   'CAFilledCylinderCenter'.
 * - Each of CACylinder, EACylinder, EBCylinder, and CBCylinder
 *   consists of 4 blocks, named 'East', 'North', 'West', and 'South',
 *   so an example of a valid block name is 'CACylinderEast'.
 * - The Block group called "Outer" consists of all the CA and CB blocks. They
 *   all border the outer boundary if `IncludeOuterSphere` is false.
 * - If `IncludeOuterSphere` is true, then there are more blocks named
 *   OuterSphereCAFilledCylinder, OuterSphereCBFilledCylinder,
 *   OuterSphereCACylinder, and OuterSphereCBCylinder.
 *   These are in a Block group called "OuterSphere",
 *   and all of these border the outer boundary.
 * - The Block group called "InnerA" consists of all the EA, and MA
 *   blocks. They all border the inner boundary "A" if
 *   `IncludeInnerSphereA` is false.
 * - If `IncludeInnerSphereA` is true, then there are new blocks
 *   InnerSphereEAFilledCylinder, InnerSphereMAFilledCylinder, and
 *   InnerSphereEACylinder. These are in a Block group called "InnerSphereA",
 *   and all of these border the inner excision boundary "A".
 * - The Block group called "InnerB" consists of all the EB, and MB
 *   blocks. They all border the inner boundary "B" if
 *   `IncludeInnerSphereB` is false.
 * - If `IncludeInnerSphereB` is true, then there are new blocks
 *   InnerSphereEBFilledCylinder, InnerSphereMBFilledCylinder, and
 *   InnerSphereEBCylinder. These are in a Block group called "InnerSphereB",
 *   and all of these border the inner excision boundary "B".
 *
 * If \f$c_A\f$ and \f$c_B\f$ are the input parameters center_A and
 * center_B, \f$r_A\f$ and \f$r_B\f$ are the input parameters radius_A and
 * radius_B, and \f$R\f$ is the outer boundary radius, we demand the
 * following restrictions on parameters:
 * - \f$c_A^0>0\f$; this is a convention to simplify the code.
 * - \f$c_B^0<0\f$; this is a convention to simplify the code.
 * - \f$|c_A^0|\le|c_B^0|\f$. We should roughly have \f$r_A c_A^0 + r_B c_B^0\f$
 *   close to zero; that is, for BBHs (where \f$r_A\f$ is roughly twice the
 *   mass of the heavier object A, and \f$r_B\f$ is roughly twice the mass
 *   of the lighter object B) the center of mass should be roughly
 *   at the origin.
 * - \f$0 < r_B < r_A\f$
 * - \f$R \ge 3(|c_A^0|-|c_B^0|)\f$; otherwise the blocks will be too compressed
 *   near the outer boundary.
 *
 */
class CylindricalBinaryCompactObject : public DomainCreator<3> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf3Maps<CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval>,
                     CoordinateMaps::UniformCylindricalEndcap,
                     CoordinateMaps::DiscreteRotation<3>>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                                    CoordinateMaps::Interval>,
                     CoordinateMaps::UniformCylindricalEndcap,
                     CoordinateMaps::DiscreteRotation<3>>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf3Maps<CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval>,
                     CoordinateMaps::UniformCylindricalFlatEndcap,
                     CoordinateMaps::DiscreteRotation<3>>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                                    CoordinateMaps::Interval>,
                     CoordinateMaps::UniformCylindricalFlatEndcap,
                     CoordinateMaps::DiscreteRotation<3>>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                                    CoordinateMaps::Interval>,
                     CoordinateMaps::UniformCylindricalSide,
                     CoordinateMaps::DiscreteRotation<3>>>;

  struct CenterA {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Grid coordinates of center for Object A, which is at x>0."};
  };
  struct CenterB {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Grid coordinates of center for Object B, which is at x<0."};
  };
  struct RadiusA {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of grid boundary around Object A."};
  };
  struct RadiusB {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of grid boundary around Object B."};
  };
  struct IncludeInnerSphereA {
    using type = bool;
    static constexpr Options::String help = {
        "Add an extra spherical layer of Blocks around Object A."};
  };
  struct IncludeInnerSphereB {
    using type = bool;
    static constexpr Options::String help = {
        "Add an extra spherical layer of Blocks around Object B."};
  };
  struct IncludeOuterSphere {
    using type = bool;
    static constexpr Options::String help = {
        "Add an extra spherical layer of Blocks inside the outer boundary."};
  };
  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {
        "Grid-coordinate radius of outer boundary."};
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level. Specify one of: a single number, a list "
        "representing [r, theta, perp], or such a list for every block in the "
        "domain. Here 'r' is the radial direction normal to the inner and "
        "outer boundaries, 'theta' is the periodic direction, and 'perp' is "
        "the third direction."};
  };
  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points. Specify one of: a single number, a "
        "list representing [r, theta, perp], or such a list for every block in "
        "the domain. Here 'r' is the radial direction normal to the inner and "
        "outer boundaries, 'theta' is the periodic direction, and 'perp' is "
        "the third direction."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  struct BoundaryConditions {
    static constexpr Options::String help = "The boundary conditions to apply.";
  };
  template <typename BoundaryConditionsBase>
  struct InnerBoundaryCondition {
    static std::string name() { return "InnerBoundary"; }
    static constexpr Options::String help =
        "Options for the inner boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static std::string name() { return "OuterBoundary"; }
    static constexpr Options::String help =
        "Options for the outer boundary conditions.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
  };

  using basic_options =
      tmpl::list<CenterA, CenterB, RadiusA, RadiusB, IncludeInnerSphereA,
                 IncludeInnerSphereB, IncludeOuterSphere, OuterRadius,
                 InitialRefinement, InitialGridPoints, TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          InnerBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "The CylindricalBinaryCompactObject domain is a general domain for "
      "two compact objects. The user must provide the (grid-frame) "
      "centers and radii of the spherical inner edge of the grid surrounding "
      "each of the two compact objects A and B."};

  CylindricalBinaryCompactObject(
      typename CenterA::type center_A, typename CenterB::type center_B,
      typename RadiusA::type radius_A, typename RadiusB::type radius_B,
      typename IncludeInnerSphereA::type include_inner_sphere_A,
      typename IncludeInnerSphereB::type include_inner_sphere_B,
      typename IncludeOuterSphere::type include_outer_sphere,
      typename OuterRadius::type outer_radius,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_grid_points,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  CylindricalBinaryCompactObject() = default;
  CylindricalBinaryCompactObject(const CylindricalBinaryCompactObject&) =
      delete;
  CylindricalBinaryCompactObject(CylindricalBinaryCompactObject&&) = default;
  CylindricalBinaryCompactObject& operator=(
      const CylindricalBinaryCompactObject&) = delete;
  CylindricalBinaryCompactObject& operator=(CylindricalBinaryCompactObject&&) =
      default;
  ~CylindricalBinaryCompactObject() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

 private:
  // Note that center_A_ and center_B_ are rotated with respect to the
  // input centers (which are in the grid frame), so that we can
  // construct the map in a frame where the centers are offset in the
  // z direction.  At the end, there will be another rotation back to
  // the grid frame (where the centers are offset in the x direction).
  typename CenterA::type center_A_{};
  typename CenterB::type center_B_{};
  typename RadiusA::type radius_A_{};
  typename RadiusB::type radius_B_{};
  typename IncludeInnerSphereA::type include_inner_sphere_A_{};
  typename IncludeInnerSphereB::type include_inner_sphere_B_{};
  typename IncludeOuterSphere::type include_outer_sphere_{};
  typename OuterRadius::type outer_radius_{};
  typename std::vector<std::array<size_t, 3>> initial_refinement_{};
  typename std::vector<std::array<size_t, 3>> initial_grid_points_{};
  // cut_spheres_offset_factor_ is eta in Eq. (A.9) of
  // https://arxiv.org/abs/1206.3015.  cut_spheres_offset_factor_
  // could be set to unity to simplify the equations.  Here we fix it
  // to the value 0.99 used in SpEC, so that we reproduce SpEC's
  // domain decomposition.
  double cut_spheres_offset_factor_{0.99};
  // z_cutting_plane_ is x_C in Eq. (A.9) of
  // https://arxiv.org/abs/1206.3015 (but rotated to the z-axis).
  double z_cutting_plane_{};
  size_t number_of_blocks_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};
}  // namespace domain::creators
