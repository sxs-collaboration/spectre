// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/RotationMap.hpp"

#include <array>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain::creators::time_dependent_options {
template <size_t NumDerivs>
RotationMapOptions<NumDerivs>::RotationMapOptions(
    std::variant<std::vector<std::array<double, 4>>,
                 FromVolumeFile<names::Rotation>>
        initial_quaternions,
    std::optional<std::vector<std::array<double, 3>>> initial_angles,
    std::optional<double> decay_timescale_in, const Options::Context& context)
    : decay_timescale(decay_timescale_in) {
  quaternions = make_array<NumDerivs + 1, DataVector>(DataVector{4, 0.0});
  angles = make_array<NumDerivs + 1, DataVector>(DataVector{3, 0.0});

  if (std::holds_alternative<std::vector<std::array<double, 4>>>(
          initial_quaternions)) {
    auto& values =
        std::get<std::vector<std::array<double, 4>>>(initial_quaternions);
    if (values.empty() or values.size() > quaternions.size()) {
      PARSE_ERROR(
          context,
          "Must specify at least the value of the quaternion, and optionally "
          "up to "
              << NumDerivs << " time derivatives.");
    }
    for (size_t i = 0; i < values.size(); i++) {
      gsl::at(quaternions, i) =
          DataVector{values[i][0], values[i][1], values[i][2], values[i][3]};
    }

    if (initial_angles.has_value()) {
      auto& angle_values = initial_angles.value();
      if (angle_values.empty() or angle_values.size() > angles.size()) {
        PARSE_ERROR(
            context,
            "When specifying the angle, you must specify at least the value, "
            "and optionally up to "
                << NumDerivs << " time derivatives.");
      }
      for (size_t i = 0; i < angle_values.size(); i++) {
        gsl::at(angles, i) = DataVector{angle_values[i][0], angle_values[i][1],
                                        angle_values[i][2]};
      }
    }
  } else if (std::holds_alternative<FromVolumeFile<names::Rotation>>(
                 initial_quaternions)) {
    if (decay_timescale.has_value()) {
      PARSE_ERROR(context,
                  "When specifying the initial quaternions from a volume file, "
                  "the decay timescale must be 'Auto'.");
    }
    auto& values_from_file =
        std::get<FromVolumeFile<names::Rotation>>(initial_quaternions);

    for (size_t i = 0; i < values_from_file.quaternions.size(); i++) {
      gsl::at(quaternions, i) = gsl::at(values_from_file.quaternions, i);
      gsl::at(angles, i) = gsl::at(values_from_file.angle_values, i);
    }

    if (initial_angles.has_value()) {
      // Reset angle func so derivs that weren't specified are zero
      angles = make_array<NumDerivs + 1, DataVector>(DataVector{3, 0.0});
      auto& angle_values = initial_angles.value();
      if (angle_values.empty() or angle_values.size() > angles.size()) {
        PARSE_ERROR(
            context,
            "When specifying the angle, you must specify at least the value, "
            "and optionally up to "
                << NumDerivs << " time derivatives.");
      }
      for (size_t i = 0; i < angle_values.size(); i++) {
        gsl::at(angles, i) = DataVector{angle_values[i][0], angle_values[i][1],
                                        angle_values[i][2]};
      }
    }
  }
}

#define NUMDERIVS(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template struct RotationMapOptions<NUMDERIVS(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#undef NUMDERIVS
}  // namespace domain::creators::time_dependent_options
