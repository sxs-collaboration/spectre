// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainHelpers.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <size_t VolumeDim>
class DiscreteRotation;
class EquatorialCompression;
template <size_t VolumeDim>
class Identity;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <size_t Dim>
class Wedge;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/*!
 * \brief Creates a 3D Domain in the shape of a hollow spherical shell
 * consisting of six wedges.
 *
 * \image html WedgeOrientations.png "The orientation of each wedge in Shell."
 */
class Shell : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial, CoordinateMaps::Wedge<3>,
      CoordinateMaps::DiscreteRotation<3>,
      CoordinateMaps::EquatorialCompression,
      CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                     CoordinateMaps::Identity<2>>>>;

  /// Options for the EquatorialCompression map
  struct EquatorialCompressionOptions {
    static constexpr Options::String help = {
        "Options for the EquatorialCompression map, used for modifying the "
        "angular distribution of gridpoints."};
    struct AspectRatio {
      using type = double;
      static constexpr Options::String help = {
          "Applies the map tan(theta') = AspectRatio * tan(theta) to the "
          "Shell.\n"
          "The angle theta is measured with respect to the polar axis, given "
          "by "
          "the choice of IndexPolarAxis. This map preserves radial distances."};
      static double lower_bound() { return 0; }
    };
    struct IndexPolarAxis {
      using type = size_t;
      static constexpr Options::String help = {
          "The index (0, 1, or 2) of the axis along which equatorial "
          "compression is applied, where 0 is x, 1 is y, and 2 is z."};
    };
    using options = tmpl::list<AspectRatio, IndexPolarAxis>;

    double aspect_ratio;
    size_t index_polar_axis;
  };

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {"Inner radius of the Shell."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius of the Shell."};
  };

  struct InitialRefinement {
    using type = size_t;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [r,angular]."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  struct UseEquiangularMap {
    using type = bool;
    static constexpr Options::String help = {
        "Use equiangular instead of equidistant coordinates."};
  };

  struct EquatorialCompression {
    using type =
        Options::Auto<EquatorialCompressionOptions, Options::AutoLabel::None>;
    static constexpr Options::String help = {
        "Options for the EquatorialCompression map."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the shell "
        "between InnerRadius and OuterRadius. They must be given in ascending "
        "order. This should be used if boundaries need to be set at specific "
        "radii. If the number but not the specific locations of the boundaries "
        "are important, use InitialRefinement instead."};
  };

  struct RadialDistribution {
    using type = std::vector<domain::CoordinateMaps::Distribution>;
    static constexpr Options::String help = {
        "Select the radial distribution of grid points in each spherical "
        "shell. The possible values are `Linear` and `Logarithmic`. There must "
        "be N+1 radial distributions specified for N radial partitions."};
    static size_t lower_bound_on_size() { return 1; }
  };

  struct WhichWedges {
    using type = ShellWedges;
    static constexpr Options::String help = {
        "Which wedges to include in the shell."};
    static constexpr type suggested_value() { return ShellWedges::All; }
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
      tmpl::list<InnerRadius, OuterRadius, InitialRefinement, InitialGridPoints,
                 UseEquiangularMap, EquatorialCompression, RadialPartitioning,
                 RadialDistribution, WhichWedges, TimeDependence>;

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
      "Creates a 3D spherical shell with 6 Blocks. `UseEquiangularMap` has\n"
      "a default value of `true` because there is no central Block in this\n"
      "domain. Equidistant coordinates are best suited to Blocks with\n"
      "Cartesian grids. However, the option is allowed for testing "
      "purposes. The `aspect_ratio` moves grid points on the shell towards\n"
      "the equator for values greater than 1.0, and towards the poles for\n"
      "positive values less than 1.0. The user may also specify the axis\n"
      "along which this compression is applied. The user may also choose to\n"
      "use only a single wedge (along the -x direction), or four wedges\n"
      "along a chosen plane, or a whole shell with additional half wedges\n"
      "along a chosen plane using the `WhichWedges` option. Using the\n"
      "RadialPartitioning "
      "option, a user may set the locations of boundaries of radial "
      "partitions, each of which will have the grid points and refinement "
      "specified from the previous options. The RadialDistribution option "
      "specifies whether the radial grid points are distributed linearly or "
      "logarithmically for each radial partition. Therefore, there must be N+1 "
      "radial distributions specified for N radial partitions. For simple "
      "h-refinement where the number but not the locations of the radial "
      "boundaries are important, the InitialRefinement option should be used "
      "instead of RadialPartitioning."};

  Shell(double inner_radius, double outer_radius, size_t initial_refinement,
        std::array<size_t, 2> initial_number_of_grid_points,
        bool use_equiangular_map = true,
        std::optional<domain::creators::Shell::EquatorialCompressionOptions>
            equatorial_compression = {},
        std::vector<double> radial_partitioning = {},
        std::vector<domain::CoordinateMaps::Distribution> radial_distribution =
            {domain::CoordinateMaps::Distribution::Linear},
        ShellWedges = ShellWedges::All,
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
            time_dependence = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            inner_boundary_condition = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            outer_boundary_condition = nullptr,
        const Options::Context& context = {});

  Shell() = default;
  Shell(const Shell&) = delete;
  Shell(Shell&&) = default;
  Shell& operator=(const Shell&) = delete;
  Shell& operator=(Shell&&) = default;
  ~Shell() override = default;

  Domain<3> create_domain() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_radius_{};
  double outer_radius_{};
  size_t initial_refinement_{};
  std::array<size_t, 2> initial_number_of_grid_points_{};
  bool use_equiangular_map_ = true;
  double aspect_ratio_ = 1.0;
  size_t index_polar_axis_ = 2;
  std::vector<double> radial_partitioning_ = {};
  std::vector<domain::CoordinateMaps::Distribution> radial_distribution_{};
  ShellWedges which_wedges_ = ShellWedges::All;
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  size_t number_of_layers_{};
  size_t blocks_per_layer_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_;
};
}  // namespace domain::creators
