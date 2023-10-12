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

namespace ah {
/*!
 * \brief Label for what a horizon find will be used for.
 */
enum class Destination { Observation, ControlSystem };
std::ostream& operator<<(std::ostream& os, Destination destination);
}  // namespace ah

template <>
struct Options::create_from_yaml<ah::Destination> {
  template <typename Metavariables>
  static ah::Destination create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
ah::Destination Options::create_from_yaml<ah::Destination>::create<void>(
    const Options::Option& options);
