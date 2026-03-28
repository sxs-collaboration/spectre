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
class CartoonSphere1D final : public DomainCreator<3> {
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
        "Inner radius of domain, which is a sphere if set to 0, otherwise a "
        "spherical shell."};
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
      "domain."};

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

  // The block groups are Block0, Block1, ..., starting with the innermost
  // Block.
  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  Domain<3> create_domain(const Options::Context& context) const;
  double inner_bound_{};
  double outer_bound_{};
  std::vector<size_t> initial_refinement_levels_{};
  std::vector<size_t> initial_num_points_{};
  std::vector<double> radial_partitioning_{};
  std::vector<CoordinateMaps::Distribution> radial_distributions_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      inner_boundary_condition_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      outer_boundary_condition_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  size_t num_blocks_{};
  std::vector<std::string> block_names_{};
  std::unordered_map<std::string, std::unordered_set<std::string>>
      block_groups_{};
};

}  // namespace domain::creators
