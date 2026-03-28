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
/// Create a 3D Domain with its computational domain being the\f$x-y\f$
/// plane. The third dimension uses a Cartoon basis with Killing vector along
/// the \f$\phi\f$ direction.
class CartoonCylinder final : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<domain::CoordinateMap<
      Frame::BlockLogical, Frame::Inertial,
      CoordinateMaps::ProductOf3Maps<CoordinateMaps::Interval,
                                     CoordinateMaps::Interval,
                                     CoordinateMaps::Identity<1>>>>;

  static std::string name() { return "CartoonCylinder"; }

  struct LowerBounds {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Lower bound in the [x,y] dimensions."};
  };

  struct UpperBounds {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Upper bound in the [x,y] dimensions."};
  };

  struct InitialRefinement {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial refinement level for the [x,y] dimensions."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr Options::String help = {
        "Initial number of grid points in the [x,y] dimensions."};
  };

  struct Distributions {
    using type = std::array<CoordinateMaps::Distribution, 2>;
    static constexpr Options::String help = {
        "Distribution of grid points in the [x,y] dimensions."};
  };

  template <typename BoundaryConditionsBase>
  struct LowerUpperBoundaryCondition {
    static constexpr Options::String help =
        "Lower and upper Boundary Conditions";
    struct LowerBC {
      using type = std::unique_ptr<BoundaryConditionsBase>;
      static constexpr Options::String help = "Lower Boundary Condition";
      static std::string name() { return "Lower"; };
    };
    struct UpperBC {
      using type = std::unique_ptr<BoundaryConditionsBase>;
      static constexpr Options::String help = "Upper Boundary Condition";
      static std::string name() { return "Upper"; };
    };
    LowerUpperBoundaryCondition(typename LowerBC::type lower_bc,
                                typename UpperBC::type upper_bc)
        : lower(std::move(lower_bc)), upper(std::move(upper_bc)){};
    LowerUpperBoundaryCondition() = default;
    std::unique_ptr<BoundaryConditionsBase> lower;
    std::unique_ptr<BoundaryConditionsBase> upper;
    using options = tmpl::list<LowerBC, UpperBC>;
  };

  template <typename BoundaryConditionsBase>
  struct BoundaryConditions {
    static constexpr Options::String help = {
        "The boundary conditions to be imposed in the x & y dimensions. "
        "Either specify one B.C. to be imposed for both "
        "lower and upper boundary or a pair 'Lower:' and 'Upper:'."};
    using type = std::array<
        std::variant<std::unique_ptr<BoundaryConditionsBase>,
                     LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
        2>;
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain. Specify `None` for no "
        "time dependant maps."};
  };

  using basic_options =
      tmpl::list<LowerBounds, UpperBounds, InitialRefinement, InitialGridPoints,
                 Distributions, TimeDependence>;

  template <typename Metavariables>
  using options = tmpl::conditional_t<
      domain::BoundaryConditions::has_boundary_conditions_base_v<
          typename Metavariables::system>,
      tmpl::push_back<
          basic_options,
          BoundaryConditions<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>>,
      basic_options>;

  static constexpr Options::String help{
      "A cylinder domain that requires/enforces axial symmetry. The "
      "computational domain is the x-y plane, with Cartoon partial derivatives "
      "being used for the z direction."};

  CartoonCylinder(
      std::array<double, 2> lower_bounds, std::array<double, 2> upper_bounds,
      std::array<size_t, 2> initial_refinement_levels,
      std::array<size_t, 2> initial_num_points,
      std::array<CoordinateMaps::Distribution, 2> distributions = {},
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::array<std::array<std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>,
                            2>,
                 2>
          boundary_conditions = {},
      const Options::Context& context = {});

  template <typename BoundaryConditionsBase>
  CartoonCylinder(
      std::array<double, 2> lower_bounds, std::array<double, 2> upper_bounds,
      std::array<size_t, 2> initial_refinement_levels,
      std::array<size_t, 2> initial_num_points,
      std::array<CoordinateMaps::Distribution, 2> distributions = {},
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
          time_dependence = nullptr,
      std::array<
          std::variant<std::unique_ptr<BoundaryConditionsBase>,
                       LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
          2>
          boundary_conditions = {},
      const Options::Context& context = {})
      : CartoonCylinder(
            lower_bounds, upper_bounds, initial_refinement_levels,
            initial_num_points,
            distributions, std::move(time_dependence),
            transform_boundary_conditions(std::move(boundary_conditions)),
            context) {}

  CartoonCylinder() = default;
  CartoonCylinder(const CartoonCylinder&) = delete;
  CartoonCylinder(CartoonCylinder&&) = default;
  CartoonCylinder& operator=(const CartoonCylinder&) = delete;
  CartoonCylinder& operator=(CartoonCylinder&&) = default;
  ~CartoonCylinder() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  std::vector<std::string> block_names() const override { return block_names_; }

  std::unordered_map<std::string, std::unordered_set<std::string>>
  block_groups() const override {
    return {{name(), {name()}}};
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  // Transforms from option-created boundary conditions to the type used in the
  // constructor
  template <typename BoundaryConditionsBase>
  static auto transform_boundary_conditions(
      std::array<
          std::variant<std::unique_ptr<BoundaryConditionsBase>,
                       LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
          2>
          boundary_conditions)
      -> std::array<
          std::array<
              std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
              2>,
          2>;

 private:
  std::array<double, 2> lower_bounds_{};
  std::array<double, 2> upper_bounds_{};
  std::array<size_t, 2> initial_refinement_levels_{};
  std::array<size_t, 2> initial_num_points_{};
  bool is_periodic_in_y_{};
  std::array<CoordinateMaps::Distribution, 2> distributions_{};
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      2>
      boundary_conditions_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  size_t num_blocks_{};
  inline static const std::vector<std::string> block_names_{name()};
};

template <typename BoundaryConditionsBase>
auto CartoonCylinder::transform_boundary_conditions(
    std::array<
        std::variant<std::unique_ptr<BoundaryConditionsBase>,
                     LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
        2>
        boundary_conditions)
    -> std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        2> {
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      2>
      result{};
  for (size_t d = 0; d < 2; ++d) {
    if (std::holds_alternative<std::unique_ptr<BoundaryConditionsBase>>(
            boundary_conditions[d])) {
      auto bc = std::move(std::get<std::unique_ptr<BoundaryConditionsBase>>(
          boundary_conditions[d]));
      gsl::at(gsl::at(result, d), 0) = bc->get_clone();
      gsl::at(gsl::at(result, d), 1) = std::move(bc);
    } else {
      auto& bc = std::get<LowerUpperBoundaryCondition<BoundaryConditionsBase>>(
          boundary_conditions[d]);
      gsl::at(gsl::at(result, d), 0) = std::move(bc.lower);
      gsl::at(gsl::at(result, d), 1) = std::move(bc.upper);
    }
  }
  return result;
}

}  // namespace domain::creators
