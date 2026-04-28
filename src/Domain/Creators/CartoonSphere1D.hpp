// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Cartoon.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Interval;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {
/// Create a 3D Domain that is topologically a line. The 2nd and 3rd
/// dimensions use Cartoon bases with Killing vectors along the \f$\theta\f$ and
/// \f$\phi\f$ directions.
class CartoonSphere1D : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      CoordinateMaps::ProductOf3Maps<CoordinateMaps::Interval,
                                     CoordinateMaps::Identity<1>,
                                     CoordinateMaps::Identity<1>>>>;

  static std::string name() { return "CartoonSphere1D"; }

  struct InnerRadius {
    using type = double;
    static constexpr Options::String help = {
        "Inner radius of domain, greater than or equal to 0. If the origin is "
        "included, the innermost element will use a 1D Zernike basis, "
        "otherwise the domain will be a spherical shell."};
  };

  struct OuterRadius {
    using type = double;
    static constexpr Options::String help = {"Outer radius of domain."};
  };

  struct InitialRadialRefinement {
    using type = std::variant<size_t, std::vector<size_t>>;
    static constexpr Options::String help = {
        "Initial refinement level for the radial direction. If one value is "
        "given, it will be applied to all blocks, or every block can be "
        "specified individually."};
  };

  struct InitialNumberOfRadialGridPoints {
    using type = std::variant<size_t, std::vector<size_t>>;
    static constexpr Options::String help = {
        "Initial number of radial grid points. If one input is given, it "
        "will be applied to all blocks, or every block can be specified "
        "individually."};
  };

  struct RadialPartitioning {
    using type = std::vector<double>;
    static constexpr Options::String help = {
        "Radial coordinates of the boundaries splitting the radial blocks, "
        "strictly between InnerRadius and OuterRadius. They must be given in "
        "ascending order."};
  };

  struct RadialDistributions {
    using type = std::variant<CoordinateMaps::Distribution,
                              std::vector<CoordinateMaps::Distribution>>;
    static constexpr Options::String help = {
        "Distribution of grid points along the radial blocks. A single input "
        "will be applied to all blocks, or every block can be specified "
        "individually, in which case for N partitions, there must be N+1 "
        "distributions."};
  };

  template <typename BoundaryConditionsBase>
  struct InnerBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the inner boundary.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct OuterBoundaryCondition {
    static constexpr Options::String help =
        "Options for the boundary conditions at the outer boundary.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  template <typename BoundaryConditionsBase>
  struct CartoonBoundaryCondition {
    static constexpr Options::String help =
        "Cartoon boundary condition will be automatically applied to "
        "internal cartoon boundaries. No user options needed.";
    using type = std::nullptr_t;
    static std::string name() { return "CartoonBC"; }
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependant maps."};
  };

  using basic_options =
      tmpl::list<InnerRadius, OuterRadius, InitialRadialRefinement,
                 InitialNumberOfRadialGridPoints, RadialPartitioning,
                 RadialDistributions, TimeDependence>;

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
      "A sphere domain that requires/enforces spherical symmetry, resulting in "
      "a 1D computational domain (the radial axis). It uses Cartoon partial "
      "derivatives for the angular directions not in the computational "
      "domain. If an element touches the x=0 axis, it uses ZernikeB1 bases and "
      "automatically applies a system's Cartoon-type boundary condition."};

  CartoonSphere1D(
      double inner_bound, double outer_bound,
      typename InitialRadialRefinement::type&& initial_refinement_levels,
      typename InitialNumberOfRadialGridPoints::type&& initial_num_points,
      std::vector<double> radial_partitioning = {},
      const typename RadialDistributions::type& radial_distributions =
          domain::CoordinateMaps::Distribution::Linear,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          cartoon_boundary_condition = nullptr,
      const Options::Context& context = {});

  CartoonSphere1D() = default;
  CartoonSphere1D(const CartoonSphere1D&) = delete;
  CartoonSphere1D(CartoonSphere1D&&) = default;
  CartoonSphere1D& operator=(const CartoonSphere1D&) = delete;
  CartoonSphere1D& operator=(CartoonSphere1D&&) = default;
  ~CartoonSphere1D() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  // The block names are Block0, Block1, ..., starting with the innermost
  // Block.
  std::vector<std::string> block_names() const override;

  // The only block group is PositiveBlocks
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  double inner_bound_{};
  double outer_bound_{};
  std::vector<size_t> initial_refinement_levels_{};
  std::vector<size_t> initial_num_points_{};
  std::vector<double> radial_partitioning_{};
  std::vector<CoordinateMaps::Distribution> radial_distributions_{};
  bool use_zernike_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      cartoon_boundary_condition_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  size_t num_blocks_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};

}  // namespace domain::creators

namespace domain::creators::detail {
/// \brief Helper struct for CartoonSphere1D options parsing so the internal
/// cartoon boundary condition at $x = 0$ is not a required argument.
///
/// \details To get the cartoon-type boundary condition from a system we need
/// access to the system's metavariables, which is only present when parsing
/// options. The point of this design is so that the input file does not require
/// a dummy value for an InnerBoundaryCondition that is always the same for a
/// given system.
///
/// This helper's constructor does not take an InnerBoundaryCondition, which
/// means if we parse a CartoonSphere1D as a CartonSphere1DOptionsHelper
/// by having an options parsing specialization (so the input file values are
/// the helper's construtor, not CartoonSphere1D's), we can detect the
/// cartoon-type boundary condition for the given system and only then call the
/// CartoonSphere1D constructor with the extra information.
struct CartoonSphere1DOptionsHelper {
  // Inherit the options template from CartoonSphere1D
  template <typename Metavariables>
  using options = typename domain::creators::CartoonSphere1D::template options<
      Metavariables>;

  using InitialRadialRefinement =
      domain::creators::CartoonSphere1D::InitialRadialRefinement;
  using InitialNumberOfRadialGridPoints =
      domain::creators::CartoonSphere1D::InitialNumberOfRadialGridPoints;
  using RadialDistributions =
      domain::creators::CartoonSphere1D::RadialDistributions;

  static constexpr Options::String help = {"CartoonSphere1DOptionsHelper"};

  // Default constructor required by Options system
  CartoonSphere1DOptionsHelper() = default;

  // Does not take cartoon BC
  CartoonSphere1DOptionsHelper(
      double inner_bound, double outer_bound,
      typename InitialRadialRefinement::type&& initial_refinement_levels,
      typename InitialNumberOfRadialGridPoints::type&& initial_num_points,
      std::vector<double> radial_partitioning = {},
      typename RadialDistributions::type&& radial_distributions =
          domain::CoordinateMaps::Distribution::Linear,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          inner_boundary_condition = nullptr,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          outer_boundary_condition = nullptr,
      Options::Context&& context = {});

  // Same members as in CartoonSphere1D; public, to be extracted
  double inner_bound_{std::numeric_limits<double>::signaling_NaN()};
  double outer_bound_{std::numeric_limits<double>::signaling_NaN()};
  typename domain::creators::CartoonSphere1D::InitialRadialRefinement::type
      initial_refinement_levels_{};
  typename domain::creators::CartoonSphere1D::InitialNumberOfRadialGridPoints::
      type initial_num_points_{};
  std::vector<double> radial_partitioning_{};
  domain::creators::CartoonSphere1D::RadialDistributions::type
      radial_distributions_{domain::CoordinateMaps::Distribution::Linear};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_{nullptr};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_{nullptr};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{nullptr};
  std::nullptr_t cartoon_boundary_condition_{
      nullptr};  // For CartoonBoundaryCondition option
  Options::Context context_{};
};
}  // namespace domain::creators::detail

// Options parsing specialization to automate CartoonSphere1D's cartoon
// boundary condition
template <>
struct Options::create_from_yaml<domain::creators::CartoonSphere1D> {
  template <typename Metavariables>
  static domain::creators::CartoonSphere1D create(
      const Options::Option& options) {
    auto helper =
        options.parse_as<domain::creators::detail::CartoonSphere1DOptionsHelper,
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

    // Construct CartoonSphere1D using the helper's parsed data + cartoon BC
    return domain::creators::CartoonSphere1D(
        helper.inner_bound_, helper.outer_bound_,
        std::move(helper.initial_refinement_levels_),
        std::move(helper.initial_num_points_),
        std::move(helper.radial_partitioning_), helper.radial_distributions_,
        std::move(helper.time_dependence_),
        std::move(helper.inner_boundary_condition_),
        std::move(helper.outer_boundary_condition_),
        std::move(cartoon_boundary_condition), helper.context_);
  }
};
