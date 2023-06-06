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
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/String.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
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
/*!
 * \brief Cubic scaling, followed by uniform rotation about the \f$z\f$ axis:
 * \f{eqnarray*}
 * x &\to& x \cos \alpha(t) - y \sin \alpha(t)\text{,} \\
 * y &\to& x \sin \alpha(t) + y \cos \alpha(t)\text{,}
 * \f}
 * where \f$\alpha(t)\f$ is a `domain::FunctionsOfTime::FunctionOfTime`. For 3
 * spatial dimensions, \f$z \to z\f$, and the rotation is implemented as a
 * product of the 2D rotation and an identity map. The rotation is undefined
 * (and therefore unimplemented here) for 1 spatial dimension.
 *
 * The expansion is done by the
 * `domain::CoordinateMaps::TimeDependent::CubicScale` map. A linear
 * radial scaling can be used by specifying the `UseLinearScaling` bool.
 *
 * For this map, the cubic scaling goes from the grid frame to the distorted
 * frame, and the rotation goes from the distorted frame to the inertial frame.
 * This was chosen as a way of testing composed maps in the distorted frame.
 */
template <size_t MeshDim>
class ScalingAndZRotation final : public TimeDependence<MeshDim> {
  static_assert(
      MeshDim > 1,
      "ScalingAndZRotation<MeshDim> undefined for MeshDim == 1");

 private:
  using Identity = domain::CoordinateMaps::Identity<1>;
  using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<2>;
  using Rotation3D =
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation, Identity>;
  using CubicScaleMap =
      domain::CoordinateMaps::TimeDependent::CubicScale<MeshDim>;

 public:
  using maps_list = tmpl::flatten<tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, CubicScaleMap>,
      tmpl::conditional_t<
          MeshDim == 2,
          tmpl::list<domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                                           Rotation>,
                     domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                           CubicScaleMap, Rotation>>,
          tmpl::list<domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                                           Rotation3D>,
                     domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                           CubicScaleMap, Rotation3D>>>>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  /// \brief The \f$x\f$-, \f$y\f$-, and \f$z\f$-velocity.
  struct AngularVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The angular velocity of the map."};
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

  using GridToInertialMap = detail::generate_coordinate_map_t<
      Frame::Grid, Frame::Inertial,
      tmpl::list<CubicScaleMap,
                 tmpl::conditional_t<MeshDim == 2, Rotation,
                                     domain::CoordinateMaps::TimeDependent::
                                         ProductOf2Maps<Rotation, Identity>>>>;
  using GridToDistortedMap =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, CubicScaleMap>;

  using DistortedToInertialMap = detail::generate_coordinate_map_t<
      Frame::Distorted, Frame::Inertial,
      tmpl::list<tmpl::conditional_t<MeshDim == 2, Rotation,
                                     domain::CoordinateMaps::TimeDependent::
                                         ProductOf2Maps<Rotation, Identity>>>>;
  using options =
      tmpl::list<InitialTime, AngularVelocity, OuterBoundary, UseLinearScaling,
                 InitialExpansion, Velocity, Acceleration>;

  static constexpr Options::String help = {
      "A spatial radial scaling followed by a rotation about the z-axis.\n"
      "The spatial radial scaling is either based on a cubic scaling or a\n"
      "simple linear scaling. If the two expansion functions of time have\n"
      "the same name then the scaling is a linear radial scaling.\n"
      "The spatially uniform rotation about is initialized with a\n"
      "constant angular velocity."};

  ScalingAndZRotation() = default;
  ~ScalingAndZRotation() override = default;
  ScalingAndZRotation(const ScalingAndZRotation&) = delete;
  ScalingAndZRotation(ScalingAndZRotation&&) = default;
  ScalingAndZRotation& operator=(const ScalingAndZRotation&) = delete;
  ScalingAndZRotation& operator=(ScalingAndZRotation&&) = default;

  ScalingAndZRotation(double initial_time, double angular_velocity,
                      double outer_boundary, bool use_linear_scaling,
                      const std::array<double, 2>& initial_expansion,
                      const std::array<double, 2>& velocity,
                      const std::array<double, 2>& acceleration);

  auto get_clone() const -> std::unique_ptr<TimeDependence<MeshDim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, MeshDim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, MeshDim>>> override;

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, MeshDim>>> override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  template <size_t LocalDim>
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const ScalingAndZRotation<LocalDim>& lhs,
                         const ScalingAndZRotation<LocalDim>& rhs);

  GridToInertialMap grid_to_inertial_map() const;
  GridToDistortedMap grid_to_distorted_map() const;
  DistortedToInertialMap distorted_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double angular_velocity_{std::numeric_limits<double>::signaling_NaN()};
  double outer_boundary_{std::numeric_limits<double>::signaling_NaN()};
  bool use_linear_scaling_{false};
  std::array<double, 2> initial_expansion_{};
  std::array<double, 2> velocity_{};
  std::array<double, 2> acceleration_{};
  // Unlike other TimeDependences, these names aren't inline static const
  // because they can potentially be changed by the run-time option
  // use_linear_scaling in the constructor
  std::array<std::string, 3> functions_of_time_names_{
      {"CubicScaleA", "CubicScaleB", "Rotation"}};
};

template <size_t Dim>
bool operator==(const ScalingAndZRotation<Dim>& lhs,
                const ScalingAndZRotation<Dim>& rhs);

template <size_t Dim>
bool operator!=(const ScalingAndZRotation<Dim>& lhs,
                const ScalingAndZRotation<Dim>& rhs);
}  // namespace domain::creators::time_dependence
