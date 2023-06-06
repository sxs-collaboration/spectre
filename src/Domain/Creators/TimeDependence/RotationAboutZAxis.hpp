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
namespace domain::CoordinateMaps::TimeDependent {
template <typename Map1, typename Map2>
class ProductOf2Maps;
}  // namespace domain::CoordinateMaps::TimeDependent
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
 * \brief A spatially uniform rotation about the \f$z\f$ axis:
 * \f{eqnarray*}
 * x &\to& x \cos \alpha(t) - y \sin \alpha(t)\text{,} \\
 * y &\to& x \sin \alpha(t) + y \cos \alpha(t)\text{,}
 * \f}
 * where \f$\alpha(t)\f$ is a `domain::FunctionsOfTime::FunctionOfTime`. For 3
 * spatial dimensions, \f$z \to z\f$, and the rotation is implemented as a
 * product of the 2D rotation and an identity map. The rotation is undefined
 * (and therefore unimplemented here) for 1 spatial dimension.
 */
template <size_t MeshDim>
class RotationAboutZAxis final : public TimeDependence<MeshDim> {
  static_assert(MeshDim > 1,
                "RotationAboutZAxis<MeshDim> undefined for MeshDim == 1");

 private:
  using Identity = domain::CoordinateMaps::Identity<1>;
  using Rotation = domain::CoordinateMaps::TimeDependent::Rotation<2>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, Rotation>,
      domain::CoordinateMap<
          Frame::Grid, Frame::Inertial,
          CoordinateMaps::TimeDependent::ProductOf2Maps<Rotation, Identity>>>;

  static constexpr size_t mesh_dim = MeshDim;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  struct InitialAngle {
    using type = double;
    static constexpr Options::String help = {"The initial angle."};
  };
  /// \brief The \f$x\f$-, \f$y\f$-, and \f$z\f$-velocity.
  struct InitialAngularVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The initial angular velocity of the map."};
  };
  struct InitialAngularAcceleration {
    using type = double;
    static constexpr Options::String help = {
        "The initial angular acceleration of the map."};
  };

  using GridToInertialMap = detail::generate_coordinate_map_t<
      Frame::Grid, Frame::Inertial,
      tmpl::list<tmpl::conditional_t<MeshDim == 2, Rotation,
                                     domain::CoordinateMaps::TimeDependent::
                                         ProductOf2Maps<Rotation, Identity>>>>;

  using options = tmpl::list<InitialTime, InitialAngle, InitialAngularVelocity,
                             InitialAngularAcceleration>;

  static constexpr Options::String help = {
      "A spatially uniform rotation about the z axis initialized with a "
      "constant angular velocity."};

  RotationAboutZAxis() = default;
  ~RotationAboutZAxis() override = default;
  RotationAboutZAxis(const RotationAboutZAxis&) = delete;
  RotationAboutZAxis(RotationAboutZAxis&&) = default;
  RotationAboutZAxis& operator=(const RotationAboutZAxis&) = delete;
  RotationAboutZAxis& operator=(RotationAboutZAxis&&) = default;

  RotationAboutZAxis(double initial_time, double initial_angle,
                     double initial_angular_velocity,
                     double initial_angular_acceleration);

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
  friend bool operator==(const RotationAboutZAxis<LocalDim>& lhs,
                         const RotationAboutZAxis<LocalDim>& rhs);

  GridToInertialMap grid_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double initial_angle_{std::numeric_limits<double>::signaling_NaN()};
  double initial_angular_velocity_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_angular_acceleration_{
      std::numeric_limits<double>::signaling_NaN()};
  inline static const std::string function_of_time_name_{"Rotation"};
};

template <size_t Dim>
bool operator==(const RotationAboutZAxis<Dim>& lhs,
                const RotationAboutZAxis<Dim>& rhs);

template <size_t Dim>
bool operator!=(const RotationAboutZAxis<Dim>& lhs,
                const RotationAboutZAxis<Dim>& rhs);
}  // namespace domain::creators::time_dependence
