// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

class Option;
template <typename T>
struct create_from_yaml;

namespace Limiters {
/// \ingroup LimitersGroup
/// \brief Possible types of the minmod slope limiter and/or troubled-cell
/// indicator.
///
/// \see Limiters::Minmod for a description and reference.
enum class MinmodType { LambdaPi1, LambdaPiN, Muscl };

std::ostream& operator<<(std::ostream& os,
                         Limiters::MinmodType minmod_type) noexcept;
}  // namespace Limiters

template <>
struct create_from_yaml<Limiters::MinmodType> {
  template <typename Metavariables>
  static Limiters::MinmodType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
Limiters::MinmodType create_from_yaml<Limiters::MinmodType>::create<void>(
    const Option& options);
