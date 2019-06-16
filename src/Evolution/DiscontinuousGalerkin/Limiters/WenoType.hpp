// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

/// \cond
class Option;
template <typename T>
struct create_from_yaml;
/// \endcond

namespace Limiters {
/// \ingroup LimitersGroup
/// \brief Possible types of the WENO limiter
///
/// \see Limiters::Weno for a description and references.
enum class WenoType { Hweno, SimpleWeno };

std::ostream& operator<<(std::ostream& os,
                         Limiters::WenoType weno_type) noexcept;
}  // namespace Limiters

template <>
struct create_from_yaml<Limiters::WenoType> {
  template <typename Metavariables>
  static Limiters::WenoType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
Limiters::WenoType create_from_yaml<Limiters::WenoType>::create<void>(
    const Option& options);
