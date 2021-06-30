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
enum class VariablesToLimit { Conserved, Characteristic };

std::ostream& operator<<(std::ostream& os,
                         VariablesToLimit vars_to_limit) noexcept;
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
