// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines enum class amr::Isotropy.

#pragma once

#include <iosfwd>

/// \cond
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace amr {

/// \ingroup AmrGroup
/// \brief Isotropy of adaptive mesh refinement
enum class Isotropy {
  Anisotropic, /**< each dimension can be refined independently */
  Isotropic    /**< all dimensions must be refined the same */
};

/// Output operator for isotropy.
std::ostream& operator<<(std::ostream& os, const Isotropy& isotropy);
}  // namespace amr

template <>
struct Options::create_from_yaml<amr::Isotropy> {
  template <typename Metavariables>
  static amr::Isotropy create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
amr::Isotropy Options::create_from_yaml<amr::Isotropy>::create<void>(
    const Options::Option& options);
