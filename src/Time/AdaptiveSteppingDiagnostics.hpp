// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

struct AdaptiveSteppingDiagnostics {
  uint64_t number_of_slabs = 0;
  uint64_t number_of_slab_size_changes = 0;
  uint64_t number_of_steps = 0;
  uint64_t number_of_step_fraction_changes = 0;
  uint64_t number_of_step_rejections = 0;

  void pup(PUP::er& p);
};

bool operator==(const AdaptiveSteppingDiagnostics& a,
                const AdaptiveSteppingDiagnostics& b);

bool operator!=(const AdaptiveSteppingDiagnostics& a,
                const AdaptiveSteppingDiagnostics& b);
