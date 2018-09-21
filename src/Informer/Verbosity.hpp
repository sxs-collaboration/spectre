// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
template <typename T>
struct create_from_yaml;
class Option;
/// \endcond

/// \ingroup LoggingGroup
/// \brief Indicates how much informative output a class should output.
enum class Verbosity { Silent, Quiet, Verbose, Debug };

std::ostream& operator<<(std::ostream& os, const Verbosity& verbosity) noexcept;

template <>
struct create_from_yaml<Verbosity> {
  static Verbosity create(const Option& options);
};
