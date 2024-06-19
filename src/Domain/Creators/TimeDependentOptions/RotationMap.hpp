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
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::creators::time_dependent_options {
/*!
 * \brief Class to be used as an option for initializing rotation map
 * coefficients.
 */
template <size_t NumDerivs>
struct RotationMapOptions {
  using type = Options::Auto<RotationMapOptions, Options::AutoLabel::None>;
  static std::string name() { return "RotationMap"; }
  static constexpr Options::String help = {
      "Options for a time-dependent rotation of the coordinates. Specify "
      "'None' to not use this map."};

  struct InitialQuaternions {
    using type = std::variant<std::vector<std::array<double, 4>>,
                              FromVolumeFile<names::Rotation>>;
    static constexpr Options::String help = {
        "Initial values for the quaternion of the rotation map. You can "
        "optionally specify its first two time derivatives. If time "
        "derivatives aren't specified, zero will be used."};
  };

  struct InitialAngles {
    using type = Options::Auto<std::vector<std::array<double, 3>>>;
    static constexpr Options::String help = {
        "Initial values for the angle of the rotation map. If 'Auto' is "
        "specified, the behavior will depend on the 'InitialQuaternions' "
        "option. If you are reading the quaternion from a volume file, 'Auto' "
        "will use the angle values from the volume file. If you are simply "
        "specifying the quaternion and (optionally) its time derivatives, then "
        "'Auto' here will set the angle and its time derivatives to zero. If "
        "values are specified for the angle and its time derivatives, then "
        "those will override anything computed from the 'InitialQuaternions' "
        "option."};
  };

  struct DecayTimescale {
    using type = Options::Auto<double>;
    static constexpr Options::String help = {
        "The timescale for how fast the rotation approaches its asymptotic "
        "value. If this is specified, a SettleToConstant function of time will "
        "be used. If 'Auto' is specified, a PiecewisePolynomial function of "
        "time will be used. Note that if you are reading the initial "
        "quaternions from a volume file, then this option must be 'Auto'"};
  };

  using options = tmpl::list<InitialQuaternions, InitialAngles, DecayTimescale>;

  RotationMapOptions() = default;
  RotationMapOptions(
      std::variant<std::vector<std::array<double, 4>>,
                   FromVolumeFile<names::Rotation>>
          initial_quaternions,
      std::optional<std::vector<std::array<double, 3>>> initial_angles,
      std::optional<double> decay_timescale_in,
      const Options::Context& context = {});

  std::array<DataVector, NumDerivs + 1> quaternions{};
  std::array<DataVector, NumDerivs + 1> angles{};
  std::optional<double> decay_timescale{};
};
}  // namespace domain::creators::time_dependent_options
