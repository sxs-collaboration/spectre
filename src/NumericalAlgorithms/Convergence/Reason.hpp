// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>

/// \cond
template <typename T>
struct create_from_yaml;
class Option;
/// \endcond

namespace Convergence {

/*!
 * \brief The reason the algorithm has converged.
 *
 * \see Convergence::Criteria
 */
enum class Reason { MaxIterations, AbsoluteResidual, RelativeResidual };

std::ostream& operator<<(std::ostream& os, const Reason& reason) noexcept;

}  // namespace Convergence

template <>
struct create_from_yaml<Convergence::Reason> {
  template <typename Metavariables>
  static Convergence::Reason create(const Option& options) {
    return create<void>(options);
  }
};
template <>
Convergence::Reason create_from_yaml<Convergence::Reason>::create<void>(
    const Option& options);
