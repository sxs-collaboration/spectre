// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
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
/// The block numbering starts at the inner square and goes clockwise, starting
/// with the eastern wedge, through consecutive shells, then repeats this
/// pattern for all layers bottom to top.
///
/// \image html Cylinder.png "The Cylinder Domain."
class Cylinder : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Equiangular,
                                         CoordinateMaps::Equiangular,
                                         CoordinateMaps::Affine>>,
      domain::CoordinateMap<
          Frame::Logical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                         CoordinateMaps::Affine>>>;

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

  struct LowerBound {
    using type = double;
    static constexpr Options::String help = {
        "z-coordinate of the base of the cylinder."};
  };

  struct UpperBound {
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
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r, theta, z]."};
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

  struct HeightPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "z-coordinates of the boundaries splitting the domain into discs "
        "between LowerBound and UpperBound."};
  };

  struct BoundaryConditions {
    static constexpr Options::String help =
        "Options for the boundary conditions";
  };

  template <typename BoundaryConditionsBase>
  struct LowerBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Lower"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the lower base of the "
        "cylinder, i.e. at the `LowerBound` in the z-direction.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct UpperBoundaryCondition {
    using group = BoundaryConditions;
    static std::string name() noexcept { return "Upper"; }
    static constexpr Options::String help =
        "The boundary condition to be imposed on the upper base of the "
        "cylinder, i.e. at the `UpperBound` in the z-direction.";
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
      tmpl::list<InnerRadius, OuterRadius, LowerBound, UpperBound>,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<
              LowerBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              UpperBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>,
              MantleBoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>>>,
          tmpl::list<IsPeriodicInZ>>,
      tmpl::list<InitialRefinement, InitialGridPoints, UseEquiangularMap,
                 RadialPartitioning, HeightPartitioning>>;

  static constexpr Options::String help{
      "Creates a right circular Cylinder with a square prism surrounded by \n"
      "wedges. \n"
      "The cylinder can be partitioned radially into multiple cylindrical \n"
      "shells as well as partitioned along the cylinder's height into \n"
      "multiple disks. Including this partitioning, the number of Blocks is \n"
      "given by (1 + 4*(1+n_s)) * (1+n_z), where n_s is the \n"
      "length of RadialPartitioning and n_z the length of \n"
      "HeightPartitioning.\n"
      "The circularity of the wedge changes from 0 to 1 within the first \n"
      "shell.\n"
      "Only one refinement level for all dimensions is currently supported.\n"
      "The number of gridpoints in each dimension can be set independently, \n"
      "except for x and y in the square where they are equal to the number \n"
      "of gridpoints along the angular dimension of the wedges.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center block."};

  Cylinder(double inner_radius, double outer_radius, double lower_bound,
           double upper_bound, bool is_periodic_in_z, size_t initial_refinement,
           std::array<size_t, 3> initial_number_of_grid_points,
           bool use_equiangular_map,
           std::vector<double> radial_partitioning = {},
           std::vector<double> height_partitioning = {},
           const Options::Context& context = {});

  Cylinder(double inner_radius, double outer_radius, double lower_bound,
           double upper_bound,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               lower_boundary_condition,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               upper_boundary_condition,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               mantle_boundary_condition,
           size_t initial_refinement,
           std::array<size_t, 3> initial_number_of_grid_points,
           bool use_equiangular_map,
           std::vector<double> radial_partitioning = {},
           std::vector<double> height_partitioning = {},
           const Options::Context& context = {});

  Cylinder() = default;
  Cylinder(const Cylinder&) = delete;
  Cylinder(Cylinder&&) noexcept = default;
  Cylinder& operator=(const Cylinder&) = delete;
  Cylinder& operator=(Cylinder&&) noexcept = default;
  ~Cylinder() noexcept override = default;

  Domain<3> create_domain() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const
      noexcept override;

 private:
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  double lower_bound_{std::numeric_limits<double>::signaling_NaN()};
  double upper_bound_{std::numeric_limits<double>::signaling_NaN()};
  bool is_periodic_in_z_{true};
  size_t initial_refinement_{};
  std::array<size_t, 3> initial_number_of_grid_points_{};
  bool use_equiangular_map_{false};
  std::vector<double> radial_partitioning_{};
  std::vector<double> height_partitioning_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      mantle_boundary_condition_{};
};
}  // namespace domain::creators
