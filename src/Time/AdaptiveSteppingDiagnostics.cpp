// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/AdaptiveSteppingDiagnostics.hpp"

#include <pup.h>

void AdaptiveSteppingDiagnostics::pup(PUP::er& p) {
  p | number_of_slabs;
  p | number_of_slab_size_changes;
  p | number_of_steps;
  p | number_of_step_fraction_changes;
  p | number_of_step_rejections;
}

bool operator==(const AdaptiveSteppingDiagnostics& a,
                const AdaptiveSteppingDiagnostics& b) {
  return a.number_of_slabs == b.number_of_slabs and
         a.number_of_slab_size_changes == b.number_of_slab_size_changes and
         a.number_of_steps == b.number_of_steps and
         a.number_of_step_fraction_changes ==
             b.number_of_step_fraction_changes and
         a.number_of_step_rejections == b.number_of_step_rejections;
}

bool operator!=(const AdaptiveSteppingDiagnostics& a,
                const AdaptiveSteppingDiagnostics& b) {
  return not(a == b);
}
