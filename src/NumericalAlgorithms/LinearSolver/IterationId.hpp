// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class IterationId.

#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace LinearSolver {

/*!
 * \ingroup LinearSolverGroup
 * \brief Identifies a step in the linear solver algorithm
 */
struct IterationId {
  size_t step_number{0};

  IterationId() = default;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept;  // NOLINT
};

bool operator==(const IterationId& a, const IterationId& b) noexcept;
bool operator!=(const IterationId& a, const IterationId& b) noexcept;
bool operator<(const IterationId& a, const IterationId& b) noexcept;
bool operator<=(const IterationId& a, const IterationId& b) noexcept;
bool operator>(const IterationId& a, const IterationId& b) noexcept;
bool operator>=(const IterationId& a, const IterationId& b) noexcept;

std::ostream& operator<<(std::ostream& s, const IterationId& id) noexcept;

size_t hash_value(const IterationId& id) noexcept;

}  // namespace LinearSolver

namespace std {
template <>
struct hash<LinearSolver::IterationId> {
  size_t operator()(const LinearSolver::IterationId& id) const noexcept;
};
}  // namespace std
