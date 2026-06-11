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
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
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

namespace domain::creators::detail {
/// Options for filling the interior of the disk with a square
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
}  // namespace domain::creators::detail

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

  struct InitialAngularRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in theta. It will be applied to all "
        "blocks because h-refinement is not supported for the ZernikeB1 "
        "basis.\n"
        "Note: the half-wedges will have their theta values decremented by "
        "one.\n"
        "Note: the inner square, if included, will have refinement set to "
        "the theta value for the center half-circle for both dimensions "
        "(decremented by one in the halved dimension)."};
  };

  struct InitialRadialRefinement {
    using type = std::variant<size_t, std::vector<size_t>>;
    static constexpr Options::String help = {
        "Initial refinement level in r. If one value is given, it will be "
        "applied to all blocks, otherwise each shell can be specificed, "
        "from innermost to outermost.\n"};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,theta]. It will be applied to all "
        "blocks because p-refinement is not supported for the ZernikeB1 "
        "basis.\n"
        "Note: if included, the inner square will have both dimensions'"
        "number of grid points set to the theta value of the surrounding "
        "wedges."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the radial shells. "
        "Leave empty for only one layer."};
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
  struct OuterBoundaryCondition {
    using type = std::unique_ptr<BoundaryConditionsBase>;
    static constexpr Options::String help = {
        "The boundary condition to impose at the outer boundary of the "
        "domain."};
  };

  struct RadialDistribution {
    using type = std::variant<CoordinateMaps::Distribution,
                              std::vector<CoordinateMaps::Distribution>>;
    static constexpr Options::String help = {
        "Distribution of grid points for the radial direction of the wedges. "
        "A single value will be applied to all shells, or every shell can be "
        "specified individually, in which case for N partitions there must be "
        "N+1 distributions. For anything but linear in the innermost shell, "
        "the center must be excised."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependant maps."};
  };

  using basic_options =
      tmpl::list<InnerRadius, OuterRadius, InitialAngularRefinement,
                 InitialRadialRefinement, InitialGridPoints, RadialPartitioning,
                 UseEquiangularMap, Interior, RadialDistribution,
                 TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
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
      "spacings in the center half-square.\n"
      "Elements touching the x=0 axis use ZernikeB1 bases in the x direction\n"
      "and automatically apply a system's Cartoon-type boundary condition.\n"
      "Angular refinement must be globally set as the required mortar\n"
      "projection has not been implemented."};

  CartoonSphere2D(
      double inner_radius, double outer_radius,
      size_t initial_angular_refinement,
      const typename InitialRadialRefinement::type& initial_radial_refinement,
      std::array<size_t, 2> initial_number_of_grid_points,
      std::vector<double> radial_partitioning, bool use_equiangular_map,
      std::variant<Excision, InnerSquare> interior,
      const typename RadialDistribution::type& radial_distribution,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          cartoon_boundary_condition,
      const Options::Context& context);

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

  /// The block names are Shell0_LowerY, Shell0_UpperX, Shell0_UpperY,
  /// Shell1_LowerY, and so on. The naming and numbering goes from outer-most
  /// shell, starting from bottom and going counterclockwise, going to inside
  /// neighboring shell. If the center is included, the "center" half-circle
  /// follows same numbering with the _HalfSquare being the last block.
  std::vector<std::string> block_names() const override { return block_names_; }

  /// The block groups are Shell0, Shell1, ... starting from the outermost
  /// partition and working in.
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
  size_t initial_angular_refinement_{};
  std::vector<size_t> initial_radial_refinement_{};
  std::array<size_t, 2> initial_number_of_grid_points_{};
  std::vector<double> radial_partitioning_{};
  bool use_equiangular_map_{false};
  std::variant<Excision, InnerSquare> interior_{};
  bool fill_interior_{false};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_ = nullptr;
  std::vector<CoordinateMaps::Distribution> radial_distributions_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_ = nullptr;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      cartoon_boundary_condition_ = nullptr;
  std::vector<std::string> block_names_;
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_;
  size_t num_shells_{};
  size_t num_blocks_{};
};
}  // namespace domain::creators

namespace domain::creators::detail {
/// \brief Helper struct for CartoonSphere2D options parsing so the internal
/// cartoon boundary condition at $x = 0$ is not a required argument.
///
/// \details To get the cartoon-type boundary condition from a system we need
/// access to the system's metavariables, which is only present when parsing
/// options. The point of this design is so that the input file does not require
/// a dummy value for an InnerBoundaryCondition that is always the same for a
/// given system.
///
/// This helper's constructor does not take an InnerBoundaryCondition, which
/// means if we parse a CartoonSphere2D as a CartonSphere2DOptionsHelper
/// by having an options parsing specialization (so the input file values are
/// the helper's construtor, not CartoonSphere2D's), we can detect the
/// cartoon-type boundary condition for the given system and only then call the
/// CartoonSphere2D constructor with the extra information.
struct CartoonSphere2DOptionsHelper {
  // Inherit the options template from CartoonSphere2D
  template <typename Metavariables>
  using options = typename domain::creators::CartoonSphere2D::template options<
      Metavariables>;

  using InitialRadialRefinement =
      domain::creators::CartoonSphere2D::InitialRadialRefinement;
  using Interior = domain::creators::CartoonSphere2D::Interior;

  template <typename BoundaryConditionsBase>
  using OuterBoundaryCondition =
      domain::creators::CartoonSphere2D::OuterBoundaryCondition<
          BoundaryConditionsBase>;

  static constexpr Options::String help = {"CartoonSphere2DOptionsHelper."};

  // Default constructor required by Options system
  CartoonSphere2DOptionsHelper() = default;

  // Does not take cartoon BC
  CartoonSphere2DOptionsHelper(
      double inner_radius, double outer_radius,
      size_t initial_angular_refinement,
      typename InitialRadialRefinement::type&& initial_radial_refinement,
      std::array<size_t, 2> initial_number_of_grid_points,
      std::vector<double> radial_partitioning, bool use_equiangular_map,
      std::variant<Excision, InnerSquare> interior,
      typename domain::creators::CartoonSphere2D::RadialDistribution::type
          radial_distribution = CoordinateMaps::Distribution::Linear,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      Options::Context&& context = {});

  // Same members as in CartoonSphere2D; public, to be extracted
  double inner_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  size_t initial_angular_refinement_{0};
  typename InitialRadialRefinement::type initial_radial_refinement_{};
  std::array<size_t, 2> initial_number_of_grid_points_{};
  std::vector<double> radial_partitioning_{};
  bool use_equiangular_map_{false};
  typename Interior::type interior_{};
  typename domain::creators::CartoonSphere2D::RadialDistribution::type
      radial_distribution_{CoordinateMaps::Distribution::Linear};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_{nullptr};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{nullptr};
  Options::Context context_{};
};
}  // namespace domain::creators::detail

// Options parsing specialization to automate CartoonSphere2D's cartoon
// boundary condition
template <>
struct Options::create_from_yaml<domain::creators::CartoonSphere2D> {
  template <typename Metavariables>
  static domain::creators::CartoonSphere2D create(
      const Options::Option& options) {
    auto helper =
        options.parse_as<domain::creators::detail::CartoonSphere2DOptionsHelper,
                         Metavariables>();

    // Create cartoon BC if system supports it, if not will throw parse error in
    // real constructor
    std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
        cartoon_boundary_condition = nullptr;
    if constexpr (domain::BoundaryConditions::has_boundary_conditions_base_v<
                      typename Metavariables::system>) {
      if constexpr (domain::BoundaryConditions::system_has_cartoon_bc_v<
                        Metavariables>) {
        cartoon_boundary_condition =
            domain::BoundaryConditions::make_cartoon_boundary_condition<
                Metavariables>();
      }
    }

    // Construct CartoonSphere2D using the helper's parsed data + cartoon BC
    return domain::creators::CartoonSphere2D(
        helper.inner_radius_, helper.outer_radius_,
        helper.initial_angular_refinement_,
        std::move(helper.initial_radial_refinement_),
        std::move(helper.initial_number_of_grid_points_),
        std::move(helper.radial_partitioning_), helper.use_equiangular_map_,
        std::move(helper.interior_), helper.radial_distribution_,
        std::move(helper.time_dependence_),
        std::move(helper.outer_boundary_condition_),
        std::move(cartoon_boundary_condition), helper.context_);
  }
};
