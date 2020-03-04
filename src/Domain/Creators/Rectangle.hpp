// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Rectangle.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

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
    static constexpr OptionString help = {
        "Sequence of [x,y] for lower bounds."};
  };

  struct UpperBound {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {
        "Sequence of [x,y] for upper bounds."};
  };
  struct IsPeriodicIn {
    using type = std::array<bool, 2>;
    static constexpr OptionString help = {
        "Sequence for [x,y], true if periodic."};
    static type default_value() noexcept { return make_array<2>(false); }
  };

  struct InitialRefinement {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {"Initial refinement level in [x,y]."};
  };

  struct InitialGridPoints {
    using type = std::array<size_t, 2>;
    static constexpr OptionString help = {
        "Initial number of grid points in [x,y]."};
  };
  struct TimeDependence {
    using type =
        std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>;
    static constexpr OptionString help = {
        "The time dependence of the moving mesh domain."};
    static type default_value() noexcept;
  };

  using options =
      tmpl::list<LowerBound, UpperBound, IsPeriodicIn, InitialRefinement,
                 InitialGridPoints, TimeDependence>;

  static constexpr OptionString help{"Creates a 2D rectangle."};

  Rectangle(
      typename LowerBound::type lower_xy, typename UpperBound::type upper_xy,
      typename IsPeriodicIn::type is_periodic_in_xy,
      typename InitialRefinement::type initial_refinement_level_xy,
      typename InitialGridPoints::type initial_number_of_grid_points_in_xy,
      std::unique_ptr<domain::creators::time_dependence::TimeDependence<2>>
          time_dependence = nullptr) noexcept;

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
      time_dependence_;
};
}  // namespace creators
}  // namespace domain
