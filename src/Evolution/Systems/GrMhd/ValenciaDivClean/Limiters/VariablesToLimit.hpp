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

namespace grmhd {
namespace ValenciaDivClean {
namespace Limiters {
/// \ingroup LimitersGroup
/// \brief Type of ValenciaDivClean system variables to apply limiter to
enum class VariablesToLimit { Conserved, NumericalCharacteristic };

std::ostream& operator<<(std::ostream& os,
                         VariablesToLimit vars_to_limit) noexcept;
}  // namespace Limiters
}  // namespace ValenciaDivClean
}  // namespace grmhd

template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::Limiters::VariablesToLimit> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::Limiters::VariablesToLimit create(
      const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
grmhd::ValenciaDivClean::Limiters::VariablesToLimit
Options::create_from_yaml<grmhd::ValenciaDivClean::Limiters::VariablesToLimit>::
    create<void>(const Options::Option& options);
