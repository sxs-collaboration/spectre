// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime

template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain {
namespace creators {
namespace time_dependence {
/// \brief A linear or cubic radial scaling time dependence.
///
/// Adds the `domain::CoordinateMaps::TimeDependent::CubicScale` map. A linear
/// radial scaling can be used by specifying the two functions of time to have
/// the same name.
template <size_t MeshDim>
class CubicScale final : public TimeDependence<MeshDim> {
 private:
  using CubicScaleMap =
      domain::CoordinateMaps::TimeDependent::CubicScale<MeshDim>;

 public:
  using maps_list =
      tmpl::list<CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr OptionString help = {
        "The initial time of the functions of time"};
  };
  /// \brief The outer boundary or pivot point of the
  /// `domain::CoordinateMaps::TimeDependent::CubicScale` map
  struct OuterBoundary {
    using type = double;
    static constexpr OptionString help = {
        "Outer boundary or pivot point of the map"};
  };
  /// \brief The initial values of the expansion factors.
  struct InitialExpansion {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {"Expansion values at initial time."};
  };
  /// \brief The velocity of the expansion factors.
  struct Velocity {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {"The rate of expansion."};
  };
  /// \brief The acceleration of the expansion factors.
  struct Acceleration {
    using type = std::array<double, 2>;
    static constexpr OptionString help = {"The acceleration of expansion."};
  };
  /// \brief The names of the functions of times to be added to the added to the
  /// DataBox.
  ///
  /// The defaults are `"ExpansionA", "ExpansionB"`.
  ///
  /// If the two names are same then a linear radial scaling is used instead of
  /// a cubic scaling.
  struct FunctionOfTimeNames {
    using type = std::array<std::string, 2>;
    static constexpr OptionString help = {"Names of the functions of time."};
    static type default_value() noexcept {
      return {{"ExpansionA", "ExpansionB"}};
    }
  };

  using options = tmpl::list<InitialTime, OuterBoundary, FunctionOfTimeNames,
                             InitialExpansion, Velocity, Acceleration>;

  static constexpr OptionString help = {
      "A spatial radial scaling either based on a cubic scaling or a simple\n"
      "linear scaling.\n"
      "\n"
      "If the two functions of time have the same name then the scaling is a\n"
      "linear radial scaling."};

  using MapForComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>;

  CubicScale() = default;
  ~CubicScale() override = default;
  CubicScale(const CubicScale&) = delete;
  CubicScale(CubicScale&&) noexcept = default;
  CubicScale& operator=(const CubicScale&) = delete;
  CubicScale& operator=(CubicScale&&) noexcept = default;

  CubicScale(double initial_time,
             double outer_boundary,
             std::array<std::string, 2> functions_of_time_names,
             const std::array<double, 2>& initial_expansion,
             const std::array<double, 2>& velocity,
             const std::array<double, 2>& acceleration) noexcept;

  auto get_clone() const noexcept
      -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps(size_t number_of_blocks) const noexcept
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time() const noexcept -> std::unordered_map<
      std::string,
      std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

  /// Returns the map for each block to be used in a composition of
  /// `TimeDependence`s.
  MapForComposition map_for_composition() const noexcept;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const CubicScale<LocalDim>& lhs,
                         const CubicScale<LocalDim>& rhs) noexcept;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double outer_boundary_{std::numeric_limits<double>::signaling_NaN()};
  std::array<std::string, 2> functions_of_time_names_{};
  std::array<double, 2> initial_expansion_{};
  std::array<double, 2> velocity_{};
  std::array<double, 2> acceleration_{};
};

template <size_t Dim>
bool operator==(const CubicScale<Dim>& lhs,
                const CubicScale<Dim>& rhs) noexcept;

template <size_t Dim>
bool operator!=(const CubicScale<Dim>& lhs,
                const CubicScale<Dim>& rhs) noexcept;
}  // namespace time_dependence
}  // namespace creators
}  // namespace domain
