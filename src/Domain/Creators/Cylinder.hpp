// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <variant>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
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
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/// Create a 3D Domain in the shape of a cylinder where the cross-section
/// is a square surrounded by four two-dimensional wedges (see `Wedge`).
///
/// The outer shell can be split into sub-shells and the cylinder can be split
/// into disks along its height.
/// The block numbering starts at the inner square and goes counter-clockwise,
/// starting with the eastern wedge (+x-direction), through consecutive shells,
/// then repeats this pattern for all layers bottom to top.
///
/// \image html Cylinder.png "The Cylinder Domain."
class Cylinder : public DomainCreator<3> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf3Maps<CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval,
                                                    CoordinateMaps::Interval>>,
                 domain::CoordinateMap<
                     Frame::BlockLogical, Frame::Inertial,
                     CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                                    CoordinateMaps::Interval>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner square."};
    static double lower_bound() noexcept { return 0.; }
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Radius of the cylinder."};
    static double lower_bound() noexcept { return 0.; }
  };

  struct LowerZBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the base of the cylinder."};
  };

  struct UpperZBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the top of the cylinder."};
  };

  struct IsPeriodicInZ {
    using type = bool;
    static constexpr Options::String help = {
        "True if periodic in the cylindrical z direction."};
  };

  struct InitialRefinement {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial refinement level. Specify one of: a single number, a list "
        "representing [r, theta, z], or such a list for every block in the "
        "domain. The central cube always uses the value for 'theta' in both "
        "x- and y-direction."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<size_t, std::array<size_t, 3>,
                     std::vector<std::array<size_t, 3>>,
                     std::unordered_map<std::string, std::array<size_t, 3>>>;
    static constexpr Options::String help = {
        "Initial number of grid points. Specify one of: a single number, a "
        "list representing [r, theta, z], or such a list for every block in "
        "the domain. The central cube always uses the value for 'theta' in "
        "both x- and y-direction."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the outer shell "
        "between InnerRadius and OuterRadius."};
  };

  struct PartitioningInZ {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "z-coordinates of the boundaries splitting the domain into layers "
        "between LowerZBound and UpperZBound."};
  };

  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each cylindrical "
        "shell. The innermost shell must have a 'Linear' distribution because "
        "it changes in circularity. The 'RadialPartitioning' determines the "
        "number of shells."};
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct DistributionInZ {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the distribution of grid points along the z-axis in each "
        "layer. The lowermost layer must have a 'Linear' distribution, because "
        "both a 'Logarithmic' and 'Inverse' distribution places its "
        "singularity at 'LowerZBound'. The 'PartitioningInZ' determines the "
        "number of layers."};
    static size_t lower_bound_on_size() noexcept { return 1; }
  };

  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  template <typename BoundaryConditionsBase>
  struct LowerZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "LowerZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower base of the "
        "cylinder, i.e. at the `LowerZBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct UpperZBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "UpperZ"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper base of the "
        "cylinder, i.e. at the `UpperZBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct MantleBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Mantle"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the mantle of the "
        "cylinder, i.e. at the `OuterRadius` in the radial direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename Metavariables>
  using options = tmpl::append<
      tmpl::list<InnerRadius, OuterRadius, LowerZBound, UpperZBound>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<
              LowerZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              UpperZBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>,
          tmpl::list<IsPeriodicInZ>>,
      tmpl::list<InitialRefinement, InitialGridPoints, UseEquiangularMap,
                 RadialPartitioning, PartitioningInZ, RadialDistribution,
                 DistributionInZ>>;

  static constexpr Options::String help{
      "Creates a right circular Cylinder with a square prism surrounded by \n"
      "wedges. \n"
      "The cylinder can be partitioned radially into multiple cylindrical \n"
      "shells as well as partitioned along the cylinder's height into \n"
      "multiple layers. Including this partitioning, the number of Blocks is \n"
      "given by (1 + 4*(1+n_s)) * (1+n_z), where n_s is the \n"
      "length of RadialPartitioning and n_z the length of \n"
      "HeightPartitioning. The block numbering starts at the inner square \n"
      "and goes counter-clockwise, starting with the eastern wedge \n"
      "(+x-direction) through consecutive shells, then repeats this pattern \n"
      "for all layers bottom to top. The wedges are named as follows: \n"
      "  +x-direction: East \n"
      "  +y-direction: North \n"
      "  -x-direction: West \n"
      "  -y-direction: South \n"
      "The circularity of the wedge changes from 0 to 1 within the first \n"
      "shell.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Cylinder(
      double inner_radius, double outer_radius, double lower_z_bound,
      double upper_z_bound, bool is_periodic_in_z,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> partitioning_in_z = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      std::vector<domain::CoordinateMaps::Distribution> distribution_in_z =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  Cylinder(
      double inner_radius, double outer_radius, double lower_z_bound,
      double upper_z_bound,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          lower_z_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          upper_z_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          mantle_boundary_condition,
      const typename InitialRefinement::type& initial_refinement,
      const typename InitialGridPoints::type& initial_number_of_grid_points,
      bool use_equiangular_map, std::vector<double> radial_partitioning = {},
      std::vector<double> partitioning_in_z = {},
      std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
          {domain::CoordinateMaps::Distribution::Linear},
      std::vector<domain::CoordinateMaps::Distribution> distribution_in_z =
          {domain::CoordinateMaps::Distribution::Linear},
      const Options::Context& context = {});

  Cylinder() = default;
  Cylinder(const Cylinder&) = delete;
  Cylinder(Cylinder&&) noexcept = default;
  Cylinder& operator=(const Cylinder&) = delete;
  Cylinder& operator=(Cylinder&&) noexcept = default;
  ~Cylinder() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels()
      const noexcept override;

 private:
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double lower_z_bound_{std::numeric_limits<double>::signaling_NaN()};
  double upper_z_bound_{std::numeric_limits<double>::signaling_NaN()};
  bool is_periodic_in_z_{true};
  std::vector<std::array<size_t, 3>> initial_refinement_{};
  std::vector<std::array<size_t, 3>> initial_number_of_grid_points_{};
  bool use_equiangular_map_{false};
  std::vector<double> radial_partitioning_{};
  std::vector<double> partitioning_in_z_{};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  std::vector<domain::CoordinateMaps::Distribution> distribution_in_z_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_z_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};
};
}  // namespace domain::creators
