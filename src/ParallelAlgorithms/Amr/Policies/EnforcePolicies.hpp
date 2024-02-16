// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

/// \cond
namespace amr {
enum class Flag;
class Policies;
}  // namespace amr
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace amr {
/// \brief Updates amr_decision so that it satisfies amr_policies
template <size_t Dim>
void enforce_policies(gsl::not_null<std::array<Flag, Dim>*> amr_decision,
                      const amr::Policies& amr_policies);
}  // namespace amr
