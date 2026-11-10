// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace NewtonianMhd::Limiters {
/// \ingroup LimitersGroup
/// \brief Type of NewtonianMhd variables to apply limiter to
///
/// \note The option `Characteristic` denotes the characteristic fields computed
/// from the analytic expression, whereas the option `NumericalCharacteristic`
/// denotes the fields as computed numerically by solving for the eigenvectors
/// of the flux Jacobian.
///
/// Initial experiments with limiting in a Reimann problem by FH suggest the
/// numerical eigenvectors can sometimes produce more accurate results than the
/// analytic ones (does the numerical solution give a better linear combination
/// of the degenerate eigenvectors?), and is not too much more expensive
/// (probably the expense of the limiter as a whole dominates). More testing is
/// needed to verify and understand this...
enum class VariablesToLimit {
  Conserved,
  Characteristic,
  NumericalCharacteristic
};

std::ostream& operator<<(std::ostream& os, VariablesToLimit vars_to_limit);
}  // namespace NewtonianMhd::Limiters

template <>
struct Options::create_from_yaml<NewtonianMhd::Limiters::VariablesToLimit> {
  template <typename Metavariables>
  static NewtonianMhd::Limiters::VariablesToLimit create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
NewtonianMhd::Limiters::VariablesToLimit
Options::create_from_yaml<NewtonianMhd::Limiters::VariablesToLimit>::create<
    void>(const Options::Option& options);
