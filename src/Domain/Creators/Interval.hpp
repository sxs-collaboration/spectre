// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Interval.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Options/Options.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
/// Create a 1D Domain consisting of a single Block.
class Interval : public DomainCreator<1> {
 public:
  using maps_list =
      tmpl::list<domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                       CoordinateMaps::Affine>>;

  struct LowerBound {
    using type = std::array<double, 1>;
    static constexpr Options::String help = {
        "Sequence of [x] for lower bounds."};
  };
  struct UpperBound {
    using type = std::array<double, 1>;
    static constexpr Options::String help = {
        "Sequence of [x] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 1>;
    static constexpr Options::String help = {
        "Sequence for [x], true if periodic."};
  };
  struct InitialRefinement {
    using type = std::array<size_t, 1>;
    static constexpr Options::String help = {
        "Initial refinement level in [x]."};
  };
  struct InitialGridPoints {
    using type = std::array<size_t, 1>;
    static constexpr Options::String help = {
        "Initial number of grid points in [x]."};
  };
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };
  struct BoundaryConditions {
    static constexpr Options::String help = "The boundary conditions to apply.";
  };
  template <typename BoundaryConditionsBase>
  struct UpperBoundaryCondition {
    static std::string name() { return "UpperBoundary"; }
    static constexpr Options::String help =
        "Options for the boundary condition applied at the upper boundary.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
  };
  template <typename BoundaryConditionsBase>
  struct LowerBoundaryCondition {
    static std::string name() { return "LowerBoundary"; }
    static constexpr Options::String help =
        "Options for the boundary condition applied at the lower boundary.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
    using group = BoundaryConditions;
  };

  using common_options =
      tmpl::list<LowerBound, UpperBound, InitialRefinement, InitialGridPoints>;

  using options_periodic = tmpl::list<IsPeriodicIn>;
  template <typename System>
  using options_boundary_conditions = tmpl::list<
      LowerBoundaryCondition<
          domain::BoundaryConditions::get_boundary_conditions_base<System>>,
      UpperBoundaryCondition<
          domain::BoundaryConditions::get_boundary_conditions_base<System>>>;

  template <typename Metavariables>
  using options = tmpl::append<
      common_options,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          options_boundary_conditions<typename Metavariables::system>,
          options_periodic>,
      tmpl::list<TimeDependence>>;

  static constexpr Options::String help = {"Creates a 1D interval."};

  Interval(std::array<double, 1> lower_x, std::array<double, 1> upper_x,
           std::array<size_t, 1> initial_refinement_level_x,
           std::array<size_t, 1> initial_number_of_grid_points_in_x,
           std::array<bool, 1> is_periodic_in_x,
           std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
               time_dependence);

  Interval(std::array<double, 1> lower_x, std::array<double, 1> upper_x,
           std::array<size_t, 1> initial_refinement_level_x,
           std::array<size_t, 1> initial_number_of_grid_points_in_x,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               lower_boundary_condition,
           std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
               upper_boundary_condition,
           std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
               time_dependence,
           const Options::Context& context = {});

  Interval() = default;
  Interval(const Interval&) = delete;
  Interval(Interval&&) = default;
  Interval& operator=(const Interval&) = delete;
  Interval& operator=(Interval&&) = default;
  ~Interval() override = default;

  Domain<1> create_domain() const override;

  std::vector<std::array<size_t, 1>> initial_extents() const override;

  std::vector<std::array<size_t, 1>> initial_refinement_levels() const override;

  auto functions_of_time() const -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  typename LowerBound::type lower_x_{};
  typename UpperBound::type upper_x_{};
  typename IsPeriodicIn::type is_periodic_in_x_{};
  typename InitialRefinement::type initial_refinement_level_x_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_x_{};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      lower_boundary_condition_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      upper_boundary_condition_;
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<1>>
      time_dependence_;
};
}  // namespace creators
}  // namespace domain
