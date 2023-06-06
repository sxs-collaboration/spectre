// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace domain {
template <typename SourceFrame, typename TargetFrame, typename... Maps>
class CoordinateMap;
}  // namespace domain

namespace Frame {
struct Distorted;
struct Grid;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::time_dependence {
/// \brief A linear or cubic radial scaling time dependence.
///
/// Adds the `domain::CoordinateMaps::TimeDependent::CubicScale` map. A linear
/// radial scaling can be used by specifying the `UseLinearScaling` bool.
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
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };
  /// \brief The outer boundary or pivot point of the
  /// `domain::CoordinateMaps::TimeDependent::CubicScale` map
  struct OuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "Outer boundary or pivot point of the map"};
  };
  /// \brief The initial values of the expansion factors.
  struct InitialExpansion {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {
        "Expansion values at initial time."};
  };
  /// \brief The velocity of the expansion factors.
  struct Velocity {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {"The rate of expansion."};
  };
  /// \brief The acceleration of the expansion factors.
  struct Acceleration {
    using type = std::array<double, 2>;
    static constexpr Options::String help = {"The acceleration of expansion."};
  };
  /// \brief Whether to use linear scaling or cubic scaling.
  struct UseLinearScaling {
    using type = bool;
    static constexpr Options::String help = {
        "Whether or not to turn on cubic scaling."};
  };

  using options = tmpl::list<InitialTime, OuterBoundary, UseLinearScaling,
                             InitialExpansion, Velocity, Acceleration>;

  static constexpr Options::String help = {
      "A spatial radial scaling either based on a cubic scaling or a simple\n"
      "linear scaling.\n"
      "\n"
      "If the two functions of time have the same name then the scaling is a\n"
      "linear radial scaling."};

  using GridToInertialMap =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>;

  CubicScale() = default;
  ~CubicScale() override = default;
  CubicScale(const CubicScale&) = delete;
  CubicScale(CubicScale&&) = default;
  CubicScale& operator=(const CubicScale&) = delete;
  CubicScale& operator=(CubicScale&&) = default;

  CubicScale(double initial_time, double outer_boundary,
             bool use_linear_scaling,
             const std::array<double, 2>& initial_expansion,
             const std::array<double, 2>& velocity,
             const std::array<double, 2>& acceleration);

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, MeshDim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Grid, Frame::Distorted, MeshDim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, MeshDim>>> override {
    using ptr_type =
        domain::CoordinateMapBase<Frame::Distorted, Frame::Inertial, MeshDim>;
    return std::vector<std::unique_ptr<ptr_type>>(number_of_blocks);
  }

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const CubicScale<LocalDim>& lhs,
                         const CubicScale<LocalDim>& rhs);

  GridToInertialMap grid_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double outer_boundary_{std::numeric_limits<double>::signaling_NaN()};
  bool use_linear_scaling_{false};
  // Unlike other TimeDependences, these names aren't inline static const
  // because they can potentially be changed by the run-time option
  // use_linear_scaling in the constructor
  std::array<std::string, 2> functions_of_time_names_{
      {"CubicScaleA", "CubicScaleB"}};
  std::array<double, 2> initial_expansion_{};
  std::array<double, 2> velocity_{};
  std::array<double, 2> acceleration_{};
};

template <size_t Dim>
bool operator==(const CubicScale<Dim>& lhs, const CubicScale<Dim>& rhs);

template <size_t Dim>
bool operator!=(const CubicScale<Dim>& lhs, const CubicScale<Dim>& rhs);
}  // namespace domain::creators::time_dependence
