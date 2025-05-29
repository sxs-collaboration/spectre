// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class that holds hard coded rotation map options from the input file.
 *
 * \details This class can also be used as an option tag with the \p type type
 * alias, `name()` function, and \p help string.
 */
template <bool AllowSettleFoTs>
struct RotationMapOptions {
  using type = Options::Auto<
      std::variant<RotationMapOptions<AllowSettleFoTs>, FromVolumeFile>,
      Options::AutoLabel::None>;
  static std::string name() { return "RotationMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent rotation of the coordinates. Specify "
      "'None' to not use this map."};

  struct InitialQuaternions {
    using type = std::vector<std::array<double, 4>>;
    static constexpr Options::String help = {
        "Initial values for the quaternion of the rotation map. You can "
        "optionally specify its first two time derivatives. If time "
        "derivatives aren't specified, zero will be used."};
  };

  struct InitialAngularVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The initial angular velocity."};
  };

  struct DecayTimescale {
    using type = double;
    static constexpr Options::String help = {
        "The timescale for how fast the rotation approaches its asymptotic "
        "value with a SettleToConstant function of time."};
  };

  using non_settle_options = tmpl::list<InitialAngularVelocity>;
  using settle_options = tmpl::list<InitialQuaternions, DecayTimescale>;

  using options = tmpl::conditional_t<
      AllowSettleFoTs,
      tmpl::list<Options::Alternatives<non_settle_options, settle_options>>,
      non_settle_options>;

  RotationMapOptions() = default;
  // Constructor for non SettleToConstant functions of time
  // NOLINTNEXTLINE(google-explicit-constructor)
  RotationMapOptions(const std::array<double, 3>& initial_angular_velocity,
                     const Options::Context& context = {});
  // Constructor for SettleToConst functions of time
  RotationMapOptions(
      const std::vector<std::array<double, 4>>& initial_quaternions,
      double decay_timescale_in, const Options::Context& context = {});

  std::array<DataVector, 3> quaternions{};
  std::array<DataVector, 4> angles{};
  std::optional<double> decay_timescale;

 private:
  void initialize_angles_and_quats();
};

/*!
 * \brief Helper function that takes the variant of the rotation map options,
 * and returns the fully constructed rotation function of time.
 *
 * \details The function of time will have a new \p initial_time and \p
 * expiration_time, unless it is read from a file and is replaying.
 */
template <bool AllowSettleFoTs>
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_rotation(
    const std::variant<RotationMapOptions<AllowSettleFoTs>, FromVolumeFile>&
        rotation_map_options,
    double initial_time, double expiration_time);
}  // namespace domain::creators::time_dependent_options
