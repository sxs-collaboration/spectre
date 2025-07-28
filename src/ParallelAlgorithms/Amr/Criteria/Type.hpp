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

namespace amr::Criteria {

/// \ingroup AmrGroup
/// \brief Type of mesh refinement
enum class Type : uint8_t {
  h, /**< Used to split or join elements */
  p  /**< used to change the extents of a Mesh in an Element */
};

/// Output operator for a Type.
std::ostream& operator<<(std::ostream& os, const Type& type);
}  // namespace amr::Criteria

template <>
struct Options::create_from_yaml<amr::Criteria::Type> {
  template <typename Metavariables>
  static amr::Criteria::Type create(const Options::Option& options) {
    return create<void>(options);
  }
};

template <>
amr::Criteria::Type
Options::create_from_yaml<amr::Criteria::Type>::create<void>(
    const Options::Option& options);
