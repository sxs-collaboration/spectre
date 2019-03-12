// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>

class Option;
template <typename T>
struct create_from_yaml;

namespace SlopeLimiters {
/// \ingroup SlopeLimitersGroup
/// \brief Possible types of the minmod slope limiter and/or troubled-cell
/// indicator.
///
/// \see SlopeLimiters::Minmod for a description and reference.
enum class MinmodType { LambdaPi1, LambdaPiN, Muscl };

std::ostream& operator<<(std::ostream& os,
                         const SlopeLimiters::MinmodType& minmod_type);
}  // namespace SlopeLimiters

template <>
struct create_from_yaml<SlopeLimiters::MinmodType> {
  template <typename Metavariables>
  static SlopeLimiters::MinmodType create(const Option& options) {
    return create<void>(options);
  }
};
template <>
SlopeLimiters::MinmodType
create_from_yaml<SlopeLimiters::MinmodType>::create<void>(
    const Option& options);
