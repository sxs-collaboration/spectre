// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>
#include <utility>

#include "Options/Options.hpp"

/// Parse complex numbers as pairs of doubles.
template <>
struct Options::create_from_yaml<std::complex<double>> {
  template <typename Metavariables>
  static std::complex<double> create(const Options::Option& options) {
    const auto pair_of_doubles = options.parse_as<std::pair<double, double>>();
    return std::complex<double>(pair_of_doubles.first, pair_of_doubles.second);
  }
};
