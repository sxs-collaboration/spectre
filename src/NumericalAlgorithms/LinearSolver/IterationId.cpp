// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearSolver/IterationId.hpp"

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <ostream>  // IWYU pragma: keep
#include <pup.h>  // IWYU pragma: keep

namespace LinearSolver {

void IterationId::pup(PUP::er& p) noexcept { p | step_number; }

bool operator==(const IterationId& a, const IterationId& b) noexcept {
  return a.step_number == b.step_number;
}
bool operator!=(const IterationId& a, const IterationId& b) noexcept {
  return not(a == b);
}

bool operator<(const IterationId& a, const IterationId& b) noexcept {
  return a.step_number < b.step_number;
}
bool operator<=(const IterationId& a, const IterationId& b) noexcept {
  return not(b < a);
}
bool operator>(const IterationId& a, const IterationId& b) noexcept {
  return b < a;
}
bool operator>=(const IterationId& a, const IterationId& b) noexcept {
  return not(a < b);
}

std::ostream& operator<<(std::ostream& s, const IterationId& id) noexcept {
  return s << id.step_number;
}

size_t hash_value(const IterationId& id) noexcept {
  size_t h = 0;
  boost::hash_combine(h, id.step_number);
  return h;
}

}  // namespace LinearSolver

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<LinearSolver::IterationId>::operator()(
    const LinearSolver::IterationId& id) const noexcept {
  return boost::hash<LinearSolver::IterationId>{}(id);
}
}  // namespace std
