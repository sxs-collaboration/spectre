// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

/// Version of local time-stepping in use
enum class LtsMode : uint8_t {
  /// Global time-stepping
  Off,
  /// Conservative local time-stepping using an LtsTimeStepper
  Conservative,
};

std::ostream& operator<<(std::ostream& os, LtsMode value);

template <>
struct Options::create_from_yaml<LtsMode> {
  template <typename Metavariables>
  static LtsMode create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
LtsMode Options::create_from_yaml<LtsMode>::create<void>(
    const Options::Option& options);
