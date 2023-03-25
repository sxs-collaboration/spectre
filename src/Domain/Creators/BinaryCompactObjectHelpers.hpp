// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Options.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

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
 * This class will also create the FunctionsOfTime needed for the binary compact
 * object domains.
 *
 * \note This struct contains no information about what blocks the time
 * dependent maps will go in.
 */
struct TimeDependentMapOptions {
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

  // Names are public because they need to be used when constructing maps in the
  // BCO domain creators themselves
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string rotation_name{"Rotation"};
  inline static const std::array<std::string, 2> size_names{{"SizeA", "SizeB"}};

 private:
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
};

}  // namespace domain::creators::bco
