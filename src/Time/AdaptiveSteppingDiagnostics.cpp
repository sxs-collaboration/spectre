// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/AdaptiveSteppingDiagnostics.hpp"

#include <pup.h>

#include "Utilities/ErrorHandling/Assert.hpp"

AdaptiveSteppingDiagnostics& AdaptiveSteppingDiagnostics::operator+=(
    const AdaptiveSteppingDiagnostics& other) {
  ASSERT(number_of_slabs == other.number_of_slabs,
         "Unequal number of slabs(" << number_of_slabs << ","
                                    << other.number_of_slabs << ")");
  ASSERT(number_of_slab_size_changes == other.number_of_slab_size_changes,
         "Unequal number of slab_size_changes("
             << number_of_slab_size_changes << ","
             << other.number_of_slab_size_changes << ")");
  number_of_steps += other.number_of_steps;
  number_of_step_fraction_changes += other.number_of_step_fraction_changes;
  number_of_step_rejections += other.number_of_step_rejections;
  return *this;
}

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
