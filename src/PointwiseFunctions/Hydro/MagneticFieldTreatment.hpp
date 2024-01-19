// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

#include "Options/Options.hpp"

namespace hydro {
/// \brief Used to specify how to handle the magnetic field.
enum MagneticFieldTreatment {
  /// Assume the magnetic field is zero
  AssumeZero,
  /// Check if the magnetic field is zero
  CheckIfZero,
  /// Assume the magnetic field is non-zero
  AssumeNonZero
};

std::ostream& operator<<(std::ostream& os, MagneticFieldTreatment t);
}  // namespace hydro

/// \cond
template <>
struct Options::create_from_yaml<hydro::MagneticFieldTreatment> {
  using type = hydro::MagneticFieldTreatment;
  template <typename Metavariables>
  static type create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
hydro::MagneticFieldTreatment
Options::create_from_yaml<hydro::MagneticFieldTreatment>::create<void>(
    const Options::Option& options);
/// \endcond
