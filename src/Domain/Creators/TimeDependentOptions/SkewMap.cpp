// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/SkewMap.hpp"

#include <array>
#include <string>
#include <variant>

#include "DataStructures/DataVector.hpp"
#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"

namespace domain::creators::time_dependent_options {
SkewMapOptions::SkewMapOptions(const std::array<double, 3>& initial_angles_y_in,
                               const std::array<double, 3>& initial_angles_z_in)
    : initial_angles_y(initial_angles_y_in),
      initial_angles_z(initial_angles_z_in) {}

std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_skew(
    const std::variant<SkewMapOptions, FromVolumeFile>& skew_map_options,
    const double initial_time, const double expiration_time) {
  const std::string name{"Skew"};
  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> result{};

  if (std::holds_alternative<FromVolumeFile>(skew_map_options)) {
    const auto& from_vol_file = std::get<FromVolumeFile>(skew_map_options);
    const auto volume_fot =
        from_vol_file.retrieve_function_of_time({name}, initial_time);

    // It must be a PiecewisePolynomial
    if (UNLIKELY(dynamic_cast<domain::FunctionsOfTime::PiecewisePolynomial<2>*>(
                     volume_fot.at(name).get()) == nullptr)) {
      ERROR_NO_TRACE(
          "Skew function of time read from volume data is not a "
          "PiecewisePolynomial<2>. Cannot use it to initialize the skew map.");
    }

    result = volume_fot.at(name)->create_at_time(initial_time, expiration_time);
  } else if (std::holds_alternative<SkewMapOptions>(skew_map_options)) {
    const auto& hard_coded_options = std::get<SkewMapOptions>(skew_map_options);

    result = std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
        initial_time,
        std::array{
            DataVector{hard_coded_options.initial_angles_y[0],
                       hard_coded_options.initial_angles_z[0]},
            DataVector{hard_coded_options.initial_angles_y[1],
                       hard_coded_options.initial_angles_z[1]},
            DataVector{hard_coded_options.initial_angles_y[2],
                       hard_coded_options.initial_angles_z[2]},
        },
        expiration_time);
  } else {
    ERROR("Unknown SkewMap.");
  }

  return result;
}
}  // namespace domain::creators::time_dependent_options
