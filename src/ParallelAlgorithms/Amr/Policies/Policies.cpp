// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

#include <pup.h>
#include <pup_stl.h>

namespace amr {
Policies::Policies(const amr::Isotropy isotropy) : isotropy_(isotropy) {}

void Policies::pup(PUP::er& p) { p | isotropy_; }

bool operator==(const Policies& lhs, const Policies& rhs) {
  return lhs.isotropy() == rhs.isotropy();
}

bool operator!=(const Policies& lhs, const Policies& rhs) {
  return not(lhs == rhs);
}

}  // namespace amr
