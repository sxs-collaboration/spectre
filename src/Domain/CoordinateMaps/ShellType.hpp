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

namespace domain::CoordinateMaps {

/*!
 * \brief Type of spherical shell: built from four (2D) or six (3D) wedges
 * ("cubed") or from a single spherical shell ("spherical").
 *
 * Used to select the shell type in the input file.
 */
enum class ShellType {
  Cubed,
  Spherical,
};

std::ostream& operator<<(std::ostream& os, ShellType shell_type);

}  // namespace domain::CoordinateMaps

template <>
struct Options::create_from_yaml<domain::CoordinateMaps::ShellType> {
  template <typename Metavariables>
  static domain::CoordinateMaps::ShellType create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
domain::CoordinateMaps::ShellType
Options::create_from_yaml<domain::CoordinateMaps::ShellType>::create<void>(
    const Options::Option& options);
