// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
namespace Options {
struct Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace importers {

/// Represents the first or last observation in a volume data file, to allow
/// specifying it in an input file without knowledge of the specific observation
/// values.
enum class ObservationSelector { First, Last };

std::ostream& operator<<(std::ostream& os, const ObservationSelector value);

}  // namespace importers

template <>
struct Options::create_from_yaml<importers::ObservationSelector> {
  template <typename Metavariables>
  static importers::ObservationSelector create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
importers::ObservationSelector
Options::create_from_yaml<importers::ObservationSelector>::create<void>(
    const Options::Option& options);
