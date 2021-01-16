// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Rectangle.

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
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {
/// Create a 2D Domain consisting of a single Block.
class Rectangle : public DomainCreator<2> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::Logical, Frame::Inertial,
      CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                     CoordinateMaps::Affine>>>;

  struct LowerBound {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Sequence of [x,y] for lower bounds."};
  };

  struct UpperBound {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Sequence of [x,y] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 2>;
    static constexpr Options::String help = {
        "Sequence for [x,y], true if periodic."};
  };

  struct InitialRefinement {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial refinement level in [x,y]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in [x,y]."};
  };
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };
  template <typename BoundaryConditionsBase>
  struct BoundaryCondition {
    static std::string name() noexcept { return "BoundaryCondition"; }
    static constexpr Options::String help =
        "The boundary condition to impose on all sides.";
    using type = std::unique_ptr<BoundaryConditionsBase>;
  };

  using common_options =
      tmpl::list<LowerBound, UpperBound, InitialRefinement, InitialGridPoints>;

  using options_periodic = tmpl::list<IsPeriodicIn>;

  template <typename Metavariables>
  using options = tmpl::append<
      common_options,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          tmpl::list<BoundaryCondition<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
          options_periodic>,
      tmpl::list<TimeDependence>>;

  static constexpr Options::String help{"Creates a 2D rectangle."};

  Rectangle(
      typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
      typename InitialRefinement::type initial_refinement_level_xy,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
      typename IsPeriodicIn::type is_periodic_in_xy,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
          time_dependence = nullptr) noexcept;

  Rectangle(
      typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
      typename InitialRefinement::type initial_refinement_level_xy,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
      std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
          boundary_condition,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
          time_dependence);

  Rectangle() = default;
  Rectangle(const Rectangle&) = delete;
  Rectangle(Rectangle&&) noexcept = default;
  Rectangle& operator=(const Rectangle&) = delete;
  Rectangle& operator=(Rectangle&&) noexcept = default;
  ~Rectangle() noexcept override = default;

  Domain<2> create_domain() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_extents() const noexcept override;

  std::vector<std::array<size_t, 2>> initial_refinement_levels() const
      noexcept override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  typename LowerBound::type lower_xy_{};
  typename UpperBound::type upper_xy_{};
  typename IsPeriodicIn::type is_periodic_in_xy_{};
  typename InitialRefinement::type initial_refinement_level_xy_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xy_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
      time_dependence_{nullptr};
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_{nullptr};
};
}  // namespace creators
}  // namespace domain
