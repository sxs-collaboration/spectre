// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"

#include <array>
#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::creators::time_dependent_options {
template <bool AllowSettleFoTs>
void RotationMapOptions<AllowSettleFoTs>::initialize_angles_and_quats() {
  quaternions = make_array<3, DataVector>(DataVector{4, 0.0});
  // Defautl to the identity quaternion
  quaternions[0][0] = 1.0;
  angles = make_array<4, DataVector>(DataVector{3, 0.0});
}

template <bool AllowSettleFoTs>
RotationMapOptions<AllowSettleFoTs>::RotationMapOptions(
    const std::array<double, 3>& initial_angular_velocity,
    const Options::Context& /*context*/) {
  initialize_angles_and_quats();

  angles[1] = DataVector{initial_angular_velocity};
}

template <bool AllowSettleFoTs>
RotationMapOptions<AllowSettleFoTs>::RotationMapOptions(
    const std::vector<std::array<double, 4>>& initial_quaternions,
    const double decay_timescale_in, const Options::Context& context)
    : decay_timescale(decay_timescale_in) {
  initialize_angles_and_quats();

  if (initial_quaternions.empty() or initial_quaternions.size() > 3) {
    PARSE_ERROR(
        context,
        "Must specify at least the value of the quaternion, and optionally "
        "up to 2 time derivatives.");
  }
  for (size_t i = 0; i < initial_quaternions.size(); i++) {
    gsl::at(quaternions, i) = DataVector{initial_quaternions[i]};
  }
}

template <bool AllowSettleFoTs>
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_rotation(
    const std::variant<RotationMapOptions<AllowSettleFoTs>, FromVolumeFile>&
        rotation_map_options,
    const double initial_time, const double expiration_time) {
  const std::string name = "Rotation";
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> result{};

  if (std::holds_alternative<FromVolumeFile>(rotation_map_options)) {
    const auto& from_vol_file = std::get<FromVolumeFile>(rotation_map_options);
    auto volume_fot =
        from_vol_file.retrieve_function_of_time({name}, initial_time);

    // It must be either a QuaternionFoT or a SettleToConstant
    const auto* quat_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::QuaternionFunctionOfTime<3>*>(
            volume_fot.at(name).get());
    const auto* settle_volume_fot =
        dynamic_cast<domain::FunctionsOfTime::SettleToConstantQuaternion*>(
            volume_fot.at(name).get());

    if (UNLIKELY(quat_volume_fot == nullptr and settle_volume_fot == nullptr)) {
      ERROR_NO_TRACE(
          "Rotation function of time read from volume data is not a "
          "QuaternionFunctionOfTime<3> or a SettleToConstantQuaternion. Cannot "
          "use it to initialize the rotation map.");
    }

    if (from_vol_file.replay()) {
      result = std::move(volume_fot.at(name));
    } else {
      result =
          volume_fot.at(name)->create_at_time(initial_time, expiration_time);
    }
  } else if (std::holds_alternative<RotationMapOptions<AllowSettleFoTs>>(
                 rotation_map_options)) {
    const auto& hard_coded_options =
        std::get<RotationMapOptions<AllowSettleFoTs>>(rotation_map_options);

    if (hard_coded_options.decay_timescale.has_value()) {
      result =
          std::make_unique<domain::FunctionsOfTime::SettleToConstantQuaternion>(
              hard_coded_options.quaternions, initial_time,
              hard_coded_options.decay_timescale.value());
    } else {
      result = std::make_unique<
          domain::FunctionsOfTime::QuaternionFunctionOfTime<3>>(
          initial_time, std::array{hard_coded_options.quaternions[0]},
          hard_coded_options.angles, expiration_time);
    }
  } else {
    ERROR("Unknown RotationMap.");
  }

  return result;
}

#define ALLOWSETTLE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                             \
  template struct RotationMapOptions<ALLOWSETTLE(data)>;                 \
  template std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>      \
  get_rotation(const std::variant<RotationMapOptions<ALLOWSETTLE(data)>, \
                                  FromVolumeFile>& rotation_map_options, \
               double initial_time, double expiration_time);

GENERATE_INSTANTIATIONS(INSTANTIATE, (true, false))

#undef INSTANTIATE
#undef ALLOWSETTLE
}  // namespace domain::creators::time_dependent_options
