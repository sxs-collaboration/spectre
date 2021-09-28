// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Options/Comparator.hpp"

#include <pup.h>
#include <pup_stl.h>
#include <string>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"

namespace Options {
Comparator::Comparator(Comparison comparison) : comparison_(comparison) {}

// NOLINTNEXTLINE(google-runtime-references)
void Comparator::pup(PUP::er& p) { p | comparison_; }

Comparator create_from_yaml<Comparator>::create_impl(const Option& options) {
  const auto name = options.parse_as<std::string>();
  if (name == "EqualTo") {
    return Comparator(Comparator::Comparison::EqualTo);
  }
  if (name == "NotEqualTo") {
    return Comparator(Comparator::Comparison::NotEqualTo);
  }
  if (name == "LessThan") {
    return Comparator(Comparator::Comparison::LessThan);
  }
  if (name == "GreaterThan") {
    return Comparator(Comparator::Comparison::GreaterThan);
  }
  if (name == "LessThanOrEqualTo") {
    return Comparator(Comparator::Comparison::LessThanOrEqualTo);
  }
  if (name == "GreaterThanOrEqualTo") {
    return Comparator(Comparator::Comparison::GreaterThanOrEqualTo);
  }
  PARSE_ERROR(options.context(),
              "Invalid comparison " << name << ".  Should be EqualTo, "
              "NotEqualTo, LessThan, GreaterThan, LessThanOrEqualTo, or "
              "GreaterThanOrEqualTo.");
}
}  // namespace Options
