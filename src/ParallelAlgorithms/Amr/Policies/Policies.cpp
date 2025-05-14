// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"

#include <pup.h>

namespace amr {
Policies::Policies(const amr::Isotropy isotropy, const amr::Limits& limits,
                   const bool enforce_two_to_one_balance_in_normal_direction,
                   const bool allow_coarsening)
    : isotropy_(isotropy),
      limits_(limits),
      enforce_two_to_one_balance_in_normal_direction_(
          enforce_two_to_one_balance_in_normal_direction),
      allow_coarsening_(allow_coarsening) {}

void Policies::pup(PUP::er& p) {
  p | isotropy_;
  p | limits_;
  p | enforce_two_to_one_balance_in_normal_direction_;
  p | allow_coarsening_;
}

bool operator==(const Policies& lhs, const Policies& rhs) {
  return lhs.isotropy() == rhs.isotropy() and lhs.limits() == rhs.limits() and
         lhs.enforce_two_to_one_balance_in_normal_direction() ==
             rhs.enforce_two_to_one_balance_in_normal_direction() and
         lhs.allow_coarsening() == rhs.allow_coarsening();
}

bool operator!=(const Policies& lhs, const Policies& rhs) {
  return not(lhs == rhs);
}

}  // namespace amr
