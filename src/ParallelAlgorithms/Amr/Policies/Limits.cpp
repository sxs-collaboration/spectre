// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"

#include <pup.h>
#include <string>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/ParseError.hpp"

namespace amr {

Limits::Limits()
    : maximum_refinement_level_(ElementId<1>::max_refinement_level),
      maximum_resolution_(
          Spectral::maximum_number_of_points<Spectral::Basis::Legendre>) {}

Limits::Limits(
    const std::optional<std::array<size_t, 2>>& refinement_level_bounds,
    const std::optional<std::array<size_t, 2>>& resolution_bounds,
    const bool error_beyond_limits, const Options::Context& context)
    : minimum_refinement_level_(refinement_level_bounds.has_value()
                                    ? refinement_level_bounds.value()[0]
                                    : 0),
      maximum_refinement_level_(refinement_level_bounds.has_value()
                                    ? refinement_level_bounds.value()[1]
                                    : ElementId<1>::max_refinement_level),
      minimum_resolution_(
          resolution_bounds.has_value() ? resolution_bounds.value()[0] : 1),
      maximum_resolution_(
          resolution_bounds.has_value()
              ? resolution_bounds.value()[1]
              : Spectral::maximum_number_of_points<Spectral::Basis::Legendre>),
      error_beyond_limits_(error_beyond_limits) {
  if (minimum_refinement_level_ > ElementId<1>::max_refinement_level) {
    PARSE_ERROR(context,
                "RefinementLevel lower bound '" +
                    std::to_string(minimum_refinement_level_) +
                    "' cannot be larger than '" +
                    std::to_string(ElementId<1>::max_refinement_level) + "'.");
  }
  if (maximum_refinement_level_ > ElementId<1>::max_refinement_level) {
    PARSE_ERROR(context,
                "RefinementLevel upper bound '" +
                    std::to_string(maximum_refinement_level_) +
                    "' cannot be larger than '" +
                    std::to_string(ElementId<1>::max_refinement_level) + "'.");
  }
  if (minimum_resolution_ < 1) {
    PARSE_ERROR(context, "NumGridPoints lower bound '" +
                             std::to_string(minimum_resolution_) +
                             "' cannot be smaller than '1'.");
  }
  if (maximum_resolution_ < 1) {
    PARSE_ERROR(context, "NumGridPoints upper bound '" +
                             std::to_string(maximum_resolution_) +
                             "' cannot be smaller than '1'.");
  }
  if (minimum_resolution_ >
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>) {
    PARSE_ERROR(
        context,
        "NumGridPoints lower bound '" + std::to_string(minimum_resolution_) +
            "' cannot be larger than '" +
            std::to_string(
                Spectral::maximum_number_of_points<Spectral::Basis::Legendre>) +
            "'.");
  }
  if (maximum_resolution_ >
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>) {
    PARSE_ERROR(
        context,
        "NumGridPoints upper bound '" + std::to_string(maximum_resolution_) +
            "' cannot be larger than '" +
            std::to_string(
                Spectral::maximum_number_of_points<Spectral::Basis::Legendre>) +
            "'.");
  }
  if (minimum_refinement_level_ > maximum_refinement_level_) {
    PARSE_ERROR(context,
                "RefinementLevel lower bound '" +
                    std::to_string(minimum_refinement_level_) +
                    "' cannot be larger than RefinementLevel upper bound '" +
                    std::to_string(maximum_refinement_level_) + "'.");
  }
  if (minimum_resolution_ > maximum_resolution_) {
    PARSE_ERROR(context,
                "NumGridPoints lower bound '" +
                    std::to_string(minimum_resolution_) +
                    "' cannot be larger than NumGridPoints upper bound '" +
                    std::to_string(maximum_resolution_) + "'.");
  }
}

Limits::Limits(size_t minimum_refinement_level, size_t maximum_refinement_level,
               size_t minimum_resolution, size_t maximum_resolution)
    : minimum_refinement_level_(minimum_refinement_level),
      maximum_refinement_level_(maximum_refinement_level),
      minimum_resolution_(minimum_resolution),
      maximum_resolution_(maximum_resolution) {
  ASSERT(minimum_refinement_level_ <= maximum_refinement_level_,
         "The minimum refinement level '" +
             std::to_string(minimum_refinement_level_) +
             "' cannot be larger than the maximum refinement level '" +
             std::to_string(maximum_refinement_level_) + "'.");
  ASSERT(minimum_resolution_ <= maximum_resolution_,
         "The minimum resolution '" + std::to_string(minimum_resolution_) +
             "' cannot be larger than the maximum resolution '" +
             std::to_string(maximum_resolution_) + "'.");
}

void Limits::pup(PUP::er& p) {
  p | minimum_refinement_level_;
  p | maximum_refinement_level_;
  p | minimum_resolution_;
  p | maximum_resolution_;
  p | error_beyond_limits_;
}

bool operator==(const Limits& lhs, const Limits& rhs) {
  return lhs.minimum_refinement_level() == rhs.minimum_refinement_level() and
         lhs.maximum_refinement_level() == rhs.maximum_refinement_level() and
         lhs.minimum_resolution() == rhs.minimum_resolution() and
         lhs.maximum_resolution() == rhs.maximum_resolution() and
         lhs.error_beyond_limits() == rhs.error_beyond_limits();
}

bool operator!=(const Limits& lhs, const Limits& rhs) {
  return not(lhs == rhs);
}
}  // namespace amr
