// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

/// \ingroup LoggingGroup
/// \brief Indicates how much informative output a class should output.
enum class Verbosity { Silent, Quiet, Verbose, Debug };

std::ostream& operator<<(std::ostream& os, const Verbosity& verbosity);

template <>
struct Options::create_from_yaml<Verbosity> {
  template <typename Metavariables>
  static Verbosity create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
Verbosity Options::create_from_yaml<Verbosity>::create<void>(
    const Options::Option& options);
