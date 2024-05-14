// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ExtractionRadius.hpp"

#include <optional>
#include <stdexcept>
#include <string>

#include "Utilities/ErrorHandling/Error.hpp"

namespace Cce {
std::string get_text_radius(const std::string& cce_data_filename) {
  const size_t r_pos = cce_data_filename.find_last_of('R');
  const size_t dot_pos = cce_data_filename.find_last_of('.');
  return cce_data_filename.substr(r_pos + 1, dot_pos - r_pos - 1);
}

std::optional<double> get_extraction_radius(
    const std::string& cce_data_filename,
    const std::optional<double>& extraction_radius, const bool error) {
  const std::string text_radius = get_text_radius(cce_data_filename);
  std::optional<double> result{};
  try {
    result = extraction_radius.has_value() ? extraction_radius.value()
                                           : std::stod(text_radius);
  } catch (const std::invalid_argument&) {
    if (error) {
      ERROR(
          "The CCE filename must encode the extraction radius as an integer "
          "between the last instance of 'R' and the last instance of '.' "
          "(SpEC CCE filename format). Provided filename: "
          << cce_data_filename);
    }
  }

  return result;
}
}  // namespace Cce
