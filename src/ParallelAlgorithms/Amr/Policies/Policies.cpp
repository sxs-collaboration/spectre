// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

#include <pup.h>

namespace amr {
Policies::Policies(const amr::Isotropy isotropy, const amr::Limits& limits)
    : isotropy_(isotropy), limits_(limits) {}

void Policies::pup(PUP::er& p) {
  p | isotropy_;
  p | limits_;
}

bool operator==(const Policies& lhs, const Policies& rhs) {
  return lhs.isotropy() == rhs.isotropy() and lhs.limits() == rhs.limits();
}

bool operator!=(const Policies& lhs, const Policies& rhs) {
  return not(lhs == rhs);
}

}  // namespace amr
