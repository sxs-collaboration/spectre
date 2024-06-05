// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

#include <pup.h>

namespace amr {
Policies::Policies(const amr::Isotropy isotropy, const amr::Limits& limits,
                   const bool enforce_two_to_one_balance_in_normal_direction)
    : isotropy_(isotropy),
      limits_(limits),
      enforce_two_to_one_balance_in_normal_direction_(
          enforce_two_to_one_balance_in_normal_direction) {}

void Policies::pup(PUP::er& p) {
  p | isotropy_;
  p | limits_;
  p | enforce_two_to_one_balance_in_normal_direction_;
}

bool operator==(const Policies& lhs, const Policies& rhs) {
  return lhs.isotropy() == rhs.isotropy() and lhs.limits() == rhs.limits() and
         lhs.enforce_two_to_one_balance_in_normal_direction() ==
             rhs.enforce_two_to_one_balance_in_normal_direction();
}

bool operator!=(const Policies& lhs, const Policies& rhs) {
  return not(lhs == rhs);
}

}  // namespace amr
