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
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/Creators/TimeDependence/GenerateCoordinateMap.hpp"
#include "Domain/Creators/TimeDependence/TimeDependence.hpp"
#include "Options/Auto.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime
namespace domain::CoordinateMaps::TimeDependent {
template <bool InertiorMap>
class SphericalCompression;
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
 * \brief A spherical compression about some center, as given by
 * domain::CoordinateMaps::TimeDependent::SphericalCompression<false>.
 *
 * \details This TimeDependence is suitable for use on a spherical shell,
 * where MinRadius and MaxRadius are the inner and outer radii of the shell,
 * respectively.
 *
 * \note The quantity stored in the FunctionOfTime is really
 * the spherical-harmonic coefficient \f$\lambda_{00}(t)\f$.  This is
 * different from the Shape map, which stores YlmSpherepack coefficients
 * \f$a_{lm}(t)\f$ and \f$b_{lm}(t)\f$ instead of \f$\lambda_{lm}(t)\f$.
 * See domain::CoordinateMaps::TimeDependent::Shape for more details.
 */
class SphericalCompression final : public TimeDependence<3> {
 private:
  using SphericalCompressionMap =
      domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            SphericalCompressionMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Distorted,
                            SphericalCompressionMap>,
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                            SphericalCompressionMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                            SphericalCompressionMap, SphericalCompressionMap>>;

  static constexpr size_t mesh_dim = 3;

  /// \brief The initial time of the function of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the function of time"};
  };
  /// \brief Minimum radius for the SphericalCompression map
  struct MinRadius {
    using type = double;
    static constexpr Options::String help = {
        "Min radius for SphericalCompression map."};
  };
  /// \brief Maximum radius for the SphericalCompression map
  struct MaxRadius {
    using type = double;
    static constexpr Options::String help = {
        "Max radius for SphericalCompression map."};
  };
  /// \brief Center for the SphericalCompression map
  struct Center {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Center for the SphericalCompression map."};
  };
  /// \brief Initial value for function of time for the spherical compression
  struct InitialValue {
    using type = double;
    static constexpr Options::String help = {
        "Spherical compression value at initial time."};
  };
  /// \brief Initial radial velocity for the function of time for the spherical
  /// compression
  struct InitialVelocity {
    using type = double;
    static constexpr Options::String help = {
        "Spherical compression initial radial velocity."};
  };
  /// \brief Initial radial acceleration for the function of time for the
  /// spherical compression
  struct InitialAcceleration {
    using type = double;
    static constexpr Options::String help = {
        "Spherical compression initial radial acceleration."};
  };

  using GridToInertialMapSimple =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Inertial,
                                        tmpl::list<SphericalCompressionMap>>;
  using GridToInertialMapCombined = detail::generate_coordinate_map_t<
      Frame::Grid, Frame::Inertial,
      tmpl::list<SphericalCompressionMap, SphericalCompressionMap>>;
  using GridToDistortedMap =
      detail::generate_coordinate_map_t<Frame::Grid, Frame::Distorted,
                                        tmpl::list<SphericalCompressionMap>>;
  using DistortedToInertialMap =
      detail::generate_coordinate_map_t<Frame::Distorted, Frame::Inertial,
                                        tmpl::list<SphericalCompressionMap>>;

  using options =
      tmpl::list<InitialTime, MinRadius, MaxRadius, Center, InitialValue,
                 InitialVelocity, InitialAcceleration>;

  static constexpr Options::String help = {"A spherical compression."};

  SphericalCompression() = default;
  ~SphericalCompression() override = default;
  SphericalCompression(const SphericalCompression&) = delete;
  SphericalCompression(SphericalCompression&&) = default;
  SphericalCompression& operator=(const SphericalCompression&) = delete;
  SphericalCompression& operator=(SphericalCompression&&) = default;

  /// If SphericalCompression is created using the constructor that
  /// takes a single (value,velocity,acceleration), then there is no
  /// distorted frame (so block_maps_grid_to_distorted() and
  /// block_maps_distorted_to_inertial() contain nullptrs), and the
  /// given params go from Frame::Grid to Frame::Inertial.
  SphericalCompression(double initial_time, double min_radius,
                       double max_radius, std::array<double, 3> center,
                       double initial_value, double initial_velocity,
                       double initial_acceleration,
                       const Options::Context& context = {});

  /// If SphericalCompression is created using the constructor that
  /// takes two triplets (value,velocity,acceleration), then the first
  /// triplet describes a map from Frame::Grid to Frame::Distorted,
  /// and the second triple describes a map that goes from
  /// Frame::Distorted to Frame::Inertial.  In this case there are
  /// also two FunctionsOfTime, one for each of the two maps.
  SphericalCompression(double initial_time, double min_radius,
                       double max_radius, std::array<double, 3> center,
                       double initial_value_grid_to_distorted,
                       double initial_velocity_grid_to_distorted,
                       double initial_acceleration_grid_to_distorted,
                       double initial_value_distorted_to_inertial,
                       double initial_velocity_distorted_to_inertial,
                       double initial_acceleration_distorted_to_inertial,
                       const Options::Context& context = {});

  auto get_clone() const -> std::unique_ptr<TimeDependence<mesh_dim>> override;

  auto block_maps_grid_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Inertial, mesh_dim>>> override;

  auto block_maps_grid_to_distorted(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Grid, Frame::Distorted, mesh_dim>>> override;

  auto block_maps_distorted_to_inertial(size_t number_of_blocks) const
      -> std::vector<std::unique_ptr<domain::CoordinateMapBase<
          Frame::Distorted, Frame::Inertial, mesh_dim>>> override;

  auto functions_of_time(const std::unordered_map<std::string, double>&
                             initial_expiration_times = {}) const
      -> std::unordered_map<
          std::string,
          std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>> override;

 private:
  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const SphericalCompression& lhs,
                         const SphericalCompression& rhs);

  GridToInertialMapSimple grid_to_inertial_map_simple() const;
  GridToInertialMapCombined grid_to_inertial_map_combined() const;
  GridToDistortedMap grid_to_distorted_map() const;
  DistortedToInertialMap distorted_to_inertial_map() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  double min_radius_{std::numeric_limits<double>::signaling_NaN()};
  double max_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<double, 3> center_{};
  // If distorted and inertial frames are equal, then
  // BLA_grid_to_distorted_ is the parameter BLA for the grid-to-inertial map,
  // and BLA_distorted_to_inertial_ is unused.
  double initial_value_grid_to_distorted_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_velocity_grid_to_distorted_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_acceleration_grid_to_distorted_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_value_distorted_to_inertial_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_velocity_distorted_to_inertial_{
      std::numeric_limits<double>::signaling_NaN()};
  double initial_acceleration_distorted_to_inertial_{
      std::numeric_limits<double>::signaling_NaN()};
  bool distorted_and_inertial_frames_are_equal_{true};
  inline static const std::string function_of_time_name_grid_to_distorted_{
      "Size"};
  inline static const std::string function_of_time_name_distorted_to_inertial_{
      "SizeDistortedToInertial"};
};

bool operator!=(const SphericalCompression& lhs,
                const SphericalCompression& rhs);
}  // namespace domain::creators::time_dependence
