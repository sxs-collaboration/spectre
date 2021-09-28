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

namespace NewtonianEuler {
namespace Limiters {
/// \ingroup LimitersGroup
/// \brief Type of NewtonianEuler variables to apply limiter to
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
}  // namespace Limiters
}  // namespace NewtonianEuler

template <>
struct Options::create_from_yaml<NewtonianEuler::Limiters::VariablesToLimit> {
  template <typename Metavariables>
  static NewtonianEuler::Limiters::VariablesToLimit create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
NewtonianEuler::Limiters::VariablesToLimit
Options::create_from_yaml<NewtonianEuler::Limiters::VariablesToLimit>::create<
    void>(const Options::Option& options);
