// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
class Option;
template <typename T>
struct create_from_yaml;
/// \endcond

namespace SlopeLimiters {
/// \ingroup LimitersGroup
/// \brief Possible types of the WENO limiter
///
/// \see SlopeLimiters::Weno for a description and references.
enum class WenoType { Hweno, SimpleWeno };

std::ostream& operator<<(std::ostream& os,
                         SlopeLimiters::WenoType weno_type) noexcept;
}  // namespace SlopeLimiters

template <>
struct create_from_yaml<SlopeLimiters::WenoType> {
  template <typename Metavariables>
  static SlopeLimiters::WenoType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
SlopeLimiters::WenoType create_from_yaml<SlopeLimiters::WenoType>::create<void>(
    const Option& options);
