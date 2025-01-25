// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/GridCenters.hpp"

#include <array>
#include <fstream>
#include <iterator>
#include <optional>
#include <regex>
#include <string>
#include <variant>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/FileSystem.hpp"

namespace domain::creators::time_dependent_options {
GridCentersOptions::GridCentersOptions() = default;

GridCentersOptions::GridCentersOptions(
    const std::string& spec_evolution_parameters_perl_file,
    const std::optional<double> in_scale_inspiral_rate_by,
    const Options::Context& context)
    : initial_values{DataVector{6}, DataVector{6}, DataVector{6, 0.0}},
      scale_inspiral_rate_by(in_scale_inspiral_rate_by) {
  if (scale_inspiral_rate_by.has_value() and
      scale_inspiral_rate_by.value() <= 0.0) {
    PARSE_ERROR(context,
                "The inspiral rate must be scaled by a positive number but got "
                    << scale_inspiral_rate_by.value());
  }

  if (file_system::check_if_file_exists(spec_evolution_parameters_perl_file)) {
    std::ifstream ifs{spec_evolution_parameters_perl_file};
    const std::string spec_parameters(std::istreambuf_iterator<char>{ifs}, {});
    ifs.close();

    std::smatch match;

    std::array<double, 3> center_a{};
    std::array<double, 3> center_b{};

    if (std::regex_search(spec_parameters, match,
                          std::regex{"@CenterNS1.*=.*\\((.*),(.*),(.*)\\);"})) {
      for (size_t i = 1; i < match.size(); ++i) {
        const std::string m{match[i]};
        gsl::at(center_b, i - 1) = std::strtod(m.c_str(), nullptr);
      }
    } else {
      PARSE_ERROR(
          context,
          "Failed to parse CenterNS1 (object B location) from SpEC file.");
    }

    if (std::regex_search(spec_parameters, match,
                          std::regex{"@CenterNS2.*=.*\\((.*),(.*),(.*)\\);"})) {
      for (size_t i = 1; i < match.size(); ++i) {
        const std::string m{match[i]};
        gsl::at(center_a, i - 1) = std::strtod(m.c_str(), nullptr);
      }
    } else {
      PARSE_ERROR(
          context,
          "Failed to parse CenterNS2 (object A location) from SpEC file.");
    }

    double adot0 = std::numeric_limits<double>::signaling_NaN();
    if (std::regex_search(spec_parameters, match,
                          std::regex{"\\$ID_adot0.*=(.*);"})) {
      const std::string m{match[1]};
      adot0 = std::strtod(m.c_str(), nullptr);
    } else {
      PARSE_ERROR(context, "Failed to parse adot0 from SpEC file.");
    }
    adot0 *= scale_inspiral_rate_by.value_or(1.0);

    const std::array<double, 3> velocity_a{adot0, 0.0, 0.0};
    const std::array<double, 3> velocity_b{-adot0, 0.0, 0.0};

    for (size_t i = 0; i < 3; ++i) {
      initial_values[0][i] = gsl::at(center_a, i);
      initial_values[0][3 + i] = gsl::at(center_b, i);
      initial_values[1][i] = gsl::at(velocity_a, i);
      initial_values[1][3 + i] = gsl::at(velocity_b, i);
    }
  } else {
    PARSE_ERROR(context, "The SpEC EvolutionParameters.perl file "
                             << spec_evolution_parameters_perl_file
                             << " does not exist.");
  }
}

std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_grid_centers(
    const GridCentersOptions& grid_centers_options,
    const double initial_time, const double expiration_time) {
  return std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
      initial_time, grid_centers_options.initial_values, expiration_time);
}
}  // namespace domain::creators::time_dependent_options
