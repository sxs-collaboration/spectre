// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options

namespace fd::reconstruction {
/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Possible types of reconstruction routines to fall back to from
 * higher-order reconstruction when using adaptive method.
 *
 * \note FD reconstructions with input arguments or options (e.g. AoWeno,
 * Wcns5z, ..) are currently not supported as fallback types. We _may_ need it
 * at some point in the future but it would require using a different option
 * parsing method (e.g. factory creation).
 *
 */
enum class FallbackReconstructorType { Minmod, MonotonisedCentral, None };

std::ostream& operator<<(
    std::ostream& os,
    fd::reconstruction::FallbackReconstructorType recons_type);
}  // namespace fd::reconstruction

template <>
struct Options::create_from_yaml<
    fd::reconstruction::FallbackReconstructorType> {
  template <typename Metavariables>
  static fd::reconstruction::FallbackReconstructorType create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
fd::reconstruction::FallbackReconstructorType
Options::create_from_yaml<fd::reconstruction::FallbackReconstructorType>::
    create<void>(const Options::Option& options);
