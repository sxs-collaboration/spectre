// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
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
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
class Interval;
template <typename Map1, typename Map2>
class ProductOf2Maps;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain::creators {

/// Create a domain consisting of a single Block in `Dim` dimensions.
template <size_t Dim>
class Rectilinear : public DomainCreator<Dim> {
 private:
  static_assert(Dim == 1 or Dim == 2 or Dim == 3,
                "Rectilinear domain is only implemented in 1, 2, or 3 "
                "dimensions.");

  using Interval = CoordinateMaps::Interval;
  using Interval2D = CoordinateMaps::ProductOf2Maps<Interval, Interval>;
  using Interval3D =
      CoordinateMaps::ProductOf3Maps<Interval, Interval, Interval>;
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          tmpl::conditional_t<
              Dim == 1, Interval,
              tmpl::conditional_t<Dim == 2, Interval2D, Interval3D>>>,
      // Previous version of the domain used the `Affine` map, so we
      // need to keep it in this list for backwards compatibility.
      domain::CoordinateMap<
          Frame::BlockLogical, Frame::Inertial,
          tmpl::conditional_t<
              Dim == 1, Affine,
              tmpl::conditional_t<Dim == 2, Affine2D, Affine3D>>>>;

  static std::string name() {
    if constexpr (Dim == 1) {
      return "Interval";
    } else if constexpr (Dim == 2) {
      return "Rectangle";
    } else {
      return "Brick";
    }
  }

  struct LowerBound {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"Lower bound in each dimension."};
  };

  struct UpperBound {
    using type = std::array<double, Dim>;
    static constexpr Options::String help = {"Upper bound in each dimension."};
  };

  struct Distribution {
    using type =
        std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>;
    static constexpr Options::String help = {
        "Distribution of grid points in each dimension"};
  };

  struct IsPeriodicIn {
    using type = std::array<bool, Dim>;
    static constexpr Options::String help = {"Periodicity in each dimension."};
  };

  struct InitialRefinement {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "Initial refinement level in each dimension."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, Dim>;
    static constexpr Options::String help = {
        "Initial number of grid points in each dimension."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
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
        "The boundary conditions to be imposed in each dimension. "
        "Either specify one B.C. to be imposed for both "
        "lower and upper boundary or a pair 'Lower:' and 'Upper:'."};
    using type = std::array<
        std::variant<std::unique_ptr<BoundaryConditionsBase>,
                     LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
        Dim>;
  };

  template <typename Metavariables>
  using options = tmpl::list<
      LowerBound, UpperBound, InitialRefinement, InitialGridPoints,
      tmpl::conditional_t<
          domain::BoundaryConditions::has_boundary_conditions_base_v<
              typename Metavariables::system>,
          BoundaryConditions<
              domain::BoundaryConditions::get_boundary_conditions_base<
                  typename Metavariables::system>>,
          IsPeriodicIn>,
      Distribution, TimeDependence>;

  static constexpr Options::String help{"A rectilinear domain."};

  Rectilinear(
      std::array<double, Dim> lower_bounds,
      std::array<double, Dim> upper_bounds,
      std::array<size_t, Dim> initial_refinement_levels,
      std::array<size_t, Dim> initial_num_points,
      std::array<bool, Dim> is_periodic = make_array<Dim>(false),
      std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
          distributions = {},
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
          time_dependence = nullptr,
      const Options::Context& context = {});

  Rectilinear(
      std::array<double, Dim> lower_bounds,
      std::array<double, Dim> upper_bounds,
      std::array<size_t, Dim> initial_refinement_levels,
      std::array<size_t, Dim> initial_num_points,
      std::array<std::array<std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>,
                            2>,
                 Dim>
          boundary_conditions,
      std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
          distributions = {},
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
          time_dependence = nullptr,
      const Options::Context& context = {});

  template <typename BoundaryConditionsBase>
  Rectilinear(
      std::array<double, Dim> lower_bounds,
      std::array<double, Dim> upper_bounds,
      std::array<size_t, Dim> initial_refinement_levels,
      std::array<size_t, Dim> initial_num_points,
      std::array<
          std::variant<std::unique_ptr<BoundaryConditionsBase>,
                       LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
          Dim>
          boundary_conditions,
      std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
          distributions = {},
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
          time_dependence = nullptr,
      const Options::Context& context = {})
      : Rectilinear(
            lower_bounds, upper_bounds, initial_refinement_levels,
            initial_num_points,
            transform_boundary_conditions(std::move(boundary_conditions)),
            distributions, std::move(time_dependence), context) {}

  Rectilinear() = default;
  Rectilinear(const Rectilinear&) = delete;
  Rectilinear(Rectilinear&&) = default;
  Rectilinear& operator=(const Rectilinear&) = delete;
  Rectilinear& operator=(Rectilinear&&) = default;
  ~Rectilinear() override = default;

  Domain<Dim> create_domain() const override;

  std::vector<DirectionMap<
      Dim, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, Dim>> initial_extents() const override;

  std::vector<std::array<size_t, Dim>> initial_refinement_levels()
      const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  std::vector<std::string> block_names() const override { return block_names_; }

  // Transforms from option-created boundary conditions to the type used in the
  // constructor
  template <typename BoundaryConditionsBase>
  static auto transform_boundary_conditions(
      std::array<
          std::variant<std::unique_ptr<BoundaryConditionsBase>,
                       LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
          Dim>
          boundary_conditions)
      -> std::array<
          std::array<
              std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
              2>,
          Dim>;

 private:
  std::array<double, Dim> lower_bounds_{};
  std::array<double, Dim> upper_bounds_{};
  std::array<CoordinateMaps::DistributionAndSingularityPosition, Dim>
      distributions_{};
  std::array<bool, Dim> is_periodic_{};
  std::array<size_t, Dim> initial_refinement_levels_{};
  std::array<size_t, Dim> initial_num_points_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<Dim>>
      time_dependence_;
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      Dim>
      boundary_conditions_{};
  inline static const std::vector<std::string> block_names_{name()};
};

template <size_t Dim>
template <typename BoundaryConditionsBase>
auto Rectilinear<Dim>::transform_boundary_conditions(
    std::array<
        std::variant<std::unique_ptr<BoundaryConditionsBase>,
                     LowerUpperBoundaryCondition<BoundaryConditionsBase>>,
        Dim>
        boundary_conditions)
    -> std::array<
        std::array<
            std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>, 2>,
        Dim> {
  std::array<
      std::array<std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>,
                 2>,
      Dim>
      result{};
  for (size_t d = 0; d < Dim; ++d) {
    if (std::holds_alternative<std::unique_ptr<BoundaryConditionsBase>>(
            boundary_conditions[d])) {
      auto bc = std::move(std::get<std::unique_ptr<BoundaryConditionsBase>>(
          boundary_conditions[d]));
      result[d][0] = bc->get_clone();
      result[d][1] = std::move(bc);
    } else {
      auto& bc = std::get<LowerUpperBoundaryCondition<BoundaryConditionsBase>>(
          boundary_conditions[d]);
      result[d][0] = std::move(bc.lower);
      result[d][1] = std::move(bc.upper);
    }
  }
  return result;
}

using Interval = Rectilinear<1>;
using Rectangle = Rectilinear<2>;
using Brick = Rectilinear<3>;

}  // namespace domain::creators
