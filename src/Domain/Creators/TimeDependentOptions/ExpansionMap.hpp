// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class that holds hard coded expansion map options from the input file.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 */
template <bool AllowSettleFoTs>
struct ExpansionMapOptions {
  using type = Options::Auto<
      std::variant<ExpansionMapOptions<AllowSettleFoTs>, FromVolumeFile>,
      Options::AutoLabel::None>;
  static std::string name() { return "ExpansionMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent expansion of the coordinates. Specify "
      "'None' to not use this map."};

  struct InitialValues {
    using type = std::array<double, 3>;
    static constexpr Options::String help =
        "Initial values for the expansion map, its velocity and acceleration.";
  };

  struct InitialValuesOuterBoundary {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Initial values for the expansion map, its velocity and acceleration "
        "at the outer boundary."};
  };

  struct DecayTimescaleOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "A timescale for how fast the outer boundary expansion approaches its "
        "asymptotic value."};
  };

  struct DecayTimescale {
    using type = double;
    static constexpr Options::String help = {
        "A timescale for how fast the expansion approaches its asymptotic "
        "value with a SettleToConstant function of time."};
  };

  struct AsymptoticVelocityOuterBoundary {
    using type = double;
    static constexpr Options::String help = {
        "The constant velocity of the outer boundary expansion."};
  };

  using common_options = tmpl::list<InitialValues, DecayTimescaleOuterBoundary>;
  using settle_options =
      tmpl::push_back<common_options, InitialValuesOuterBoundary,
                      DecayTimescale>;
  using non_settle_options =
      tmpl::push_back<common_options, AsymptoticVelocityOuterBoundary>;

  using options = tmpl::conditional_t<
      AllowSettleFoTs,
      tmpl::list<Options::Alternatives<settle_options, non_settle_options>>,
      non_settle_options>;

  ExpansionMapOptions() = default;
  // Constructor for SettleToConstant functions of time
  ExpansionMapOptions(
      const std::array<double, 3>& initial_values_in,
      double decay_timescale_outer_boundary_in,
      const std::array<double, 3>& initial_values_outer_boundary_in,
      double decay_timescale_in, const Options::Context& context = {});
  // Constructor for non SettleToConstant functions of time
  ExpansionMapOptions(const std::array<double, 3>& initial_values_in,
                      double decay_timescale_outer_boundary_in,
                      double asymptotic_velocity_outer_boundary_in,
                      const Options::Context& context = {});

  std::array<DataVector, 3> initial_values{};
  std::array<DataVector, 3> initial_values_outer_boundary{};
  double decay_timescale_outer_boundary{};
  std::optional<double> decay_timescale;
  std::optional<double> asymptotic_velocity_outer_boundary;
};

/*!
 * \brief Helper functions that take the variant of the expansion map options,
 * and return the fully constructed expansion functions of time.
 *
 * \details The function of time will have a new \p initial_time and \p
 * expiration_time, unless it is read from a file and is replaying.
 */
template <bool AllowSettleFoTs>
FunctionsOfTimeMap get_expansion(
    const std::variant<ExpansionMapOptions<AllowSettleFoTs>, FromVolumeFile>&
        expansion_map_options,
    double initial_time, double expiration_time);
}  // namespace domain::creators::time_dependent_options
