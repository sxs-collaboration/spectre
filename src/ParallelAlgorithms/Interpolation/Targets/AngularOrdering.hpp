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

namespace intrp {
/*!
 * \brief Label for the ordering of spherical harmonic points on a sphere
 *
 * \details `%Strahlkorper` refers to points on a sphere ordered by SPHEREPACK
 * because `Strahlkorper`s hold YlmSpherePacks internally. `%Cce` refers to
 * points on a sphere ordered by Libsharp because Cce uses Libsharp internally.
 */
enum class AngularOrdering { Strahlkorper, Cce };
std::ostream& operator<<(std::ostream& os, AngularOrdering ordering);
}  // namespace intrp

template <>
struct Options::create_from_yaml<intrp::AngularOrdering> {
  template <typename Metavariables>
  static intrp::AngularOrdering create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
intrp::AngularOrdering
Options::create_from_yaml<intrp::AngularOrdering>::create<void>(
    const Options::Option& options);
