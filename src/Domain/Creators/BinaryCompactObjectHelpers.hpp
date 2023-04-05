// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/SphericalCompression.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

/// Namespace used to hold things used in both the BinaryCompactObject and
/// CylindricalBinaryCompactObject domain creators.
namespace domain::creators::bco {
// If `Metavariables` has a `domain` member struct and
// `domain::enable_time_dependent_maps` is `true`, then
// inherit from `std::true_type`; otherwise, inherit from `std::false_type`.
template <typename Metavariables, typename = std::void_t<>>
struct enable_time_dependent_maps : std::false_type {};

template <typename Metavariables>
struct enable_time_dependent_maps<Metavariables,
                                  std::void_t<typename Metavariables::domain>>
    : std::bool_constant<Metavariables::domain::enable_time_dependent_maps> {};

template <typename Metavariables>
constexpr bool enable_time_dependent_maps_v =
    enable_time_dependent_maps<Metavariables>::value;

/*!
 * \brief This holds all options related to the time dependent maps of the
 * binary compact object domains.
 *
 * \details Since both domains will have the same (overall) time dependent maps,
 * their options are going to be the same as well.
 *
 * This class will create the FunctionsOfTime needed for the binary compact
 * object domains as well as the actual CoordinateMap%s themselves
 *
 * \note This struct contains no information about what blocks the time
 * dependent maps will go in.
 */
struct TimeDependentMapOptions {
 private:
  template <typename SourceFrame, typename TargetFrame>
  using MapType =
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>;
  using IdentityMap = domain::CoordinateMaps::Identity<3>;
  // Time-dependent maps
  using CubicScaleMap = domain::CoordinateMaps::TimeDependent::CubicScale<3>;
  using RotationMap3D = domain::CoordinateMaps::TimeDependent::Rotation<3>;
  using CompressionMap =
      domain::CoordinateMaps::TimeDependent::SphericalCompression<false>;

  template <typename SourceFrame, typename TargetFrame>
  using IdentityForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, IdentityMap>;
  template <typename SourceFrame, typename TargetFrame>
  using CubicScaleMapForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, CubicScaleMap>;
  template <typename SourceFrame, typename TargetFrame>
  using RotationMapForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, RotationMap3D>;
  template <typename SourceFrame, typename TargetFrame>
  using CubicScaleAndRotationMapForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, CubicScaleMap,
                            RotationMap3D>;
  template <typename SourceFrame, typename TargetFrame>
  using CompressionMapForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, CompressionMap>;
  using EverythingMapForComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap,
                            CubicScaleMap, RotationMap3D>;
  using EverythingMapNoDistortedForComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, IdentityMap,
                            CubicScaleMap, RotationMap3D>;

 public:
  using maps_list = tmpl::list<
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap,
                            RotationMap3D>,
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, CompressionMap>,
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, IdentityMap>,
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial, CubicScaleMap,
                            RotationMap3D>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CompressionMap,
                            CubicScaleMap, RotationMap3D>,
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, IdentityMap,
                            CubicScaleMap, RotationMap3D>>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  /// \brief Options for the expansion map.
  /// The outer boundary radius of the map is always set to
  /// the outer boundary of the Domain, so there is no option
  /// here to set the outer boundary radius.
  struct ExpansionMapOptions {
    static constexpr Options::String help = {"Options for the expansion map."};
    struct InitialValues {
      using type = std::array<double, 2>;
      static constexpr Options::String help = {
          "Initial value and deriv of expansion."};
    };
    struct AsymptoticVelocityOuterBoundary {
      using type = double;
      static constexpr Options::String help = {
          "The asymptotic velocity of the outer boundary."};
    };
    struct DecayTimescaleOuterBoundaryVelocity {
      using type = double;
      static constexpr Options::String help = {
          "The timescale for how fast the outer boundary velocity approaches "
          "its asymptotic value."};
    };
    using options = tmpl::list<InitialValues, AsymptoticVelocityOuterBoundary,
                               DecayTimescaleOuterBoundaryVelocity>;

    std::array<double, 2> initial_values{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double outer_boundary_velocity{
        std::numeric_limits<double>::signaling_NaN()};
    double outer_boundary_decay_time{
        std::numeric_limits<double>::signaling_NaN()};
  };

  struct ExpansionMap {
    using type = ExpansionMapOptions;
    static constexpr Options::String help = {"Options for CubicScale map."};
  };

  struct RotationMap {
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about an arbitrary axis."};
  };
  struct InitialAngularVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The initial angular velocity."};
    using group = RotationMap;
  };

  template <domain::ObjectLabel Object>
  struct SizeMap {
    static std::string name() { return "SizeMap" + get_output(Object); }
    static constexpr Options::String help = {
        "Options for a time-dependent size map about the specified object."};
  };

  template <domain::ObjectLabel Object>
  struct SizeMapInitialValues {
    static std::string name() { return "InitialValues"; }
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Initial value and two derivatives of the size map."};
    using group = SizeMap<Object>;
  };

  using options = tmpl::list<InitialTime, ExpansionMap, InitialAngularVelocity,
                             SizeMapInitialValues<domain::ObjectLabel::A>,
                             SizeMapInitialValues<domain::ObjectLabel::B>>;
  static constexpr Options::String help{
      "The options for all time dependent maps in a binary compact object "
      "domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(double initial_time,
                          ExpansionMapOptions expansion_map_options,
                          std::array<double, 3> initial_angular_velocity,
                          std::array<double, 3> initial_size_values_A,
                          std::array<double, 3> initial_size_values_B);

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Expansion: `PiecewisePolynomial<2>`
   * - ExpansionOuterBoundary: `FixedSpeedCubic`
   * - Rotation: `QuaternionFunctionOfTime<3>`
   * - SizeA/B: `PiecewisePolynomial<3>`
   */
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_functions_of_time(const std::unordered_map<std::string, double>&
                               initial_expiration_times) const;

  /*!
   * \brief Construct the actual maps that will be used.
   *
   * Currently, this constructs a:
   *
   * - Expansion: `CubicScale<3>`
   * - Rotation: `Rotation<3>`
   * - SizeA/B: `SphericalCompression`
   *
   * If the inner/outer radii for an object are `std::nullopt`, this means that
   * a Size map is not constructed for that object. An identity map will be used
   * instead.
   */
  void build_maps(
      const std::array<std::array<double, 3>, 2>& centers,
      const std::array<std::optional<double>, 2>& object_inner_radii,
      const std::array<std::optional<double>, 2>& object_outer_radii,
      double domain_outer_radius);

  /*!
   * \brief This will make a composition of an `CubicScale` and `Rotation` map
   * from the templated frame `SourceFrame` to the `Frame::Inertial` frame.
   */
  template <typename SourceFrame>
  MapType<SourceFrame, Frame::Inertial> frame_to_inertial_map() const;

  /*!
   * \brief This will construct the maps from the `Frame::Grid` to the
   * `Frame::Distorted`.
   *
   * If the argument `use_identity` is true, then this will be an identity map.
   * If it is false, this currently adds a `SphericalCompression` map for the
   * templated `Object`.
   */
  template <domain::ObjectLabel Object>
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      bool use_identity) const;

  /*!
   * \brief This will construct the entire map from the `Frame::Grid` to the
   * `Frame::Inertial`.
   *
   * If the argument `include_distorted` is true, then this map will have a
   * `SphericalCompression` map from the `Frame::Grid` to the `Frame::Distorted`
   * frame. If it is false, there no grid to distorted maps will be added
   * (effectively the identity between the grid and distorted frame).
   */
  template <domain::ObjectLabel Object>
  MapType<Frame::Grid, Frame::Inertial> everything_grid_to_inertial_map(
      bool include_distorted_map) const;

  // Names are public because they need to be used when constructing maps in the
  // BCO domain creators themselves
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string rotation_name{"Rotation"};
  inline static const std::array<std::string, 2> size_names{{"SizeA", "SizeB"}};

 private:
  template <domain::ObjectLabel Object>
  size_t get_index() const;

  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  ExpansionMapOptions expansion_map_options_{};
  std::array<double, 3> initial_angular_velocity_{
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN(),
      std::numeric_limits<double>::signaling_NaN()};
  std::array<std::array<double, 3>, 2> initial_size_values_{
      std::array{std::numeric_limits<double>::signaling_NaN(),
                 std::numeric_limits<double>::signaling_NaN(),
                 std::numeric_limits<double>::signaling_NaN()},
      std::array{std::numeric_limits<double>::signaling_NaN(),
                 std::numeric_limits<double>::signaling_NaN(),
                 std::numeric_limits<double>::signaling_NaN()}};
  // Maps
  CubicScaleMap expansion_map_{};
  RotationMap3D rotation_map_{};
  std::array<CompressionMap, 2> size_maps_{};
};

}  // namespace domain::creators::bco
