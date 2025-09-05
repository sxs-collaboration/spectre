// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Equiangular;
template <size_t Dim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
template <size_t Dim>
class Wedge;
template <size_t Dim>
class DiscreteRotation;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators::detail{

/// Options for filling the interior of the sphere with a cube
struct InnerSquare {
  static constexpr Options::String help = {
      "Fill the interior of the 2D sphere with a half-square."};
  struct Sphericity {
    static std::string name() { return "FillWithSphericity"; }
    using type = double;
    static constexpr Options::String help = {
        "Sphericity of the inner half-square. Only a sphericity of "
        "0.0 is currently implemented."};
    static double lower_bound() { return 0.0; }
    static double upper_bound() { return 0.0; }
  };
  using options = tmpl::list<Sphericity>;
  InnerSquare() = default;
  explicit InnerSquare(double sphericity_in) : sphericity(sphericity_in) {}
  double sphericity = std::numeric_limits<double>::signaling_NaN();
};
} // namespace domain::creators::detail

namespace domain::creators {
/// Create a 3D Domain with a half-disk computational domain employing axial
/// symmetry. The third dimension uses a Cartoon basis with Killing vector
/// along the $\phi$ direction.
class CartoonSphere2D : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<
              CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                             CoordinateMaps::Affine>,
              CoordinateMaps::Identity<1>>>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::ProductOf2Maps<
              CoordinateMaps::ProductOf2Maps<CoordinateMaps::Equiangular,
                                             CoordinateMaps::Equiangular>,
              CoordinateMaps::Identity<1>>>,
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          CoordinateMaps::DiscreteRotation<3>,
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Wedge<2>,
                                         CoordinateMaps::Identity<1>>>>;

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Radius of the circle circumscribing the inner half-square, or the "
        "radius of the inner boundary if ExciseCenter = true."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius of the half-disk."};
  };

  struct InitialRefinement {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial refinement level in [r, theta]. If one pair is given, it will "
        "be applied to all blocks, otherwise each shell can be specified.\n"
        "Note: the half-wedges will have their theta values decremented by "
        "one.\n"
        "Note: the inner square, if included, will have refinement set to "
        "the theta value for the center half-circle for both dimensions "
        "(decremented by one in the halved dimension)."};
  };

  struct InitialGridPoints {
    using type =
        std::variant<std::array<size_t, 2>, std::vector<std::array<size_t, 2>>>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,theta]. If one pair is given, it "
        "will be applied to all blocks, otherwise each shell can be "
        "specified, from innermost to outermost.\n"
        "Note: if included, the inner square will have both dimensions'"
        "number of grid points set to the theta value of the surrounding "
        "wedges."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the radial shells. "
        "Leave emtpy for only one layer."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates in wedges."};
  };

  using Excision = detail::Excision;
  using InnerSquare = detail::InnerSquare;

  struct Interior {
    using type = std::variant<Excision, InnerSquare>;
    static constexpr Options::String help = {
        "Specify 'ExciseWithBoundaryCondition' and a boundary condition to "
        "excise the interior of the sphere, leaving a spherical shell "
        "(or just 'Excise' if boundary conditions are disabled). "
        "Or specify 'FillWithSphericity' to fill the interior."};
  };

  template <typename BoundaryConditionsBase>
  struct YAxisBoundaryCondition {
    using type = std::unique_ptr<BoundaryConditionsBase>;
    static constexpr Options::String help = {
      "The boundary condition to impose at the x=z=0 boundary."};
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    using type = std::unique_ptr<BoundaryConditionsBase>;
    static constexpr Options::String help = {
        "The boundary condition to impose at the outer boundary of the "
        "domain."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependant maps."};
  };

  using basic_options =
      tmpl::list<InnerRadius, OuterRadius, InitialRefinement, InitialGridPoints,
                 RadialPartitioning, UseEquiangularMap, Interior,
                 TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          YAxisBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          OuterBoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "Creates a sphere that requires/enforces axial-symmetry.\n"
      "The computational domain is a 2D half-disk, consisting of an inner\n"
      "half-square with two half-wedges above and below and a full wedge to\n"
      "the right. This can be extended to have mulitple shells of wedges by\n"
      "specifying a radial partition. The inner half-square can be excised,\n"
      "in which case the inner sphericity is set to 1.\n"
      "Equiangular coordinates give better gridpoint spacings in the angular\n"
      "direction, while equidistant coordinates give better gridpoint\n"
      "spacings in the center half-square."};

  CartoonSphere2D(
      double inner_radius, double outer_radius,
      typename InitialRefinement::type&& initial_refinement,
      typename InitialGridPoints::type&& initial_number_of_grid_points,
      std::vector<double> radial_partitioning, bool use_equiangular_map,
      std::variant<Excision, InnerSquare> interior,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          y_axis_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      const Options::Context& context = {});

  CartoonSphere2D() = default;
  CartoonSphere2D(const CartoonSphere2D&) = delete;
  CartoonSphere2D(CartoonSphere2D&&) = default;
  CartoonSphere2D& operator=(const CartoonSphere2D&) = delete;
  CartoonSphere2D& operator=(CartoonSphere2D&&) = default;
  ~CartoonSphere2D() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return block_groups_;
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_radius_{};
  double outer_radius_{};
  std::vector<std::array<size_t, 2>> initial_refinement_{};
  std::vector<std::array<size_t, 2>> initial_number_of_grid_points_{};
  std::vector<double> radial_partitioning_{};
  bool use_equiangular_map_{false};
  std::variant<Excision, InnerSquare> interior_{};
  bool fill_interior_{false};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      y_axis_boundary_condition_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_ = nullptr;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_shells_{};
  size_t num_blocks_{};
};
}  // namespace domain::creators
