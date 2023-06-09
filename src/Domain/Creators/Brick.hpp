// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Brick.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/GetBoundaryConditionsBase.hpp"
#include "Domain/Creators/DomainCreator.hpp"  // IWYU pragma: keep
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace CoordinateMaps {
class Affine;
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps;
}  // namespace CoordinateMaps

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain
/// \endcond

namespace domain {
namespace creators {

/// Create a 3D Domain consisting of a single Block.
class Brick : public DomainCreator<3> {
 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                            CoordinateMaps::ProductOf3Maps<
                                CoordinateMaps::Affine, CoordinateMaps::Affine,
                                CoordinateMaps::Affine>>>;

  struct LowerBound {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Sequence of [x,y,z] for lower bounds."};
  };

  struct UpperBound {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Sequence of [x,y,z] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 3>;
    static constexpr Options::String help = {
        "Sequence for [x,y,z], true if periodic."};
  };

  struct InitialRefinement {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial refinement level in [x,y,z]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 3>;
    static constexpr Options::String help = {
        "Initial number of grid points in [x,y,z]."};
  };

  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>;
    static constexpr Options::String help = {
        "The time dependence of the moving mesh domain."};
  };

  template <typename BoundaryConditionsBase, size_t Dim>
  struct BoundaryCondition {
    static std::string name() {
      return "BoundaryConditionIn" +
             std::string{Dim == 0 ? 'X' : (Dim == 1 ? 'Y' : 'Z')};
    }
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
          tmpl::list<
              BoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>,
                  0>,
              BoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>,
                  1>,
              BoundaryCondition<
                  domain::BoundaryConditions::get_boundary_conditions_base<
                      typename Metavariables::system>,
                  2>>,
          options_periodic>,
      tmpl::list<TimeDependence>>;

  static constexpr Options::String help{"Creates a 3D brick."};

  Brick(typename LowerBound::type lower_xyz,
        typename UpperBound::type upper_xyz,
        typename InitialRefinement::type initial_refinement_level_xyz,
        typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
        typename IsPeriodicIn::type is_periodic_in_xyz,
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
            time_dependence = nullptr);

  Brick(typename LowerBound::type lower_xyz,
        typename UpperBound::type upper_xyz,
        typename InitialRefinement::type initial_refinement_level_xyz,
        typename InitialGridPoints::type initial_number_of_grid_points_in_xyz,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            boundary_condition_in_x = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            boundary_condition_in_y = nullptr,
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            boundary_condition_in_z = nullptr,
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
            time_dependence = nullptr,
        const Options::Context& context = {});

  Brick() = default;
  Brick(const Brick&) = delete;
  Brick(Brick&&) = default;
  Brick& operator=(const Brick&) = delete;
  Brick& operator=(Brick&&) = default;
  ~Brick() override = default;

  Domain<3> create_domain() const override;

  std::vector<DirectionMap<
      3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>
  external_boundary_conditions() const override;

  std::vector<std::array<size_t, 3>> initial_extents() const override;

  std::vector<std::array<size_t, 3>> initial_refinement_levels() const override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  std::vector<std::string> block_names() const override { return block_names_; }

 private:
  typename LowerBound::type lower_xyz_{};
  typename UpperBound::type upper_xyz_{};
  typename IsPeriodicIn::type is_periodic_in_xyz_{};
  typename InitialRefinement::type initial_refinement_level_xyz_{};
  typename InitialGridPoints::type initial_number_of_grid_points_in_xyz_{};
  std::unique_ptr<domain::creators::time_dependence::TimeDependence<3>>
      time_dependence_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_in_x_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_in_y_;
  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition_in_z_;
  inline static const std::vector<std::string> block_names_{"Brick"};
};
}  // namespace creators
}  // namespace domain
