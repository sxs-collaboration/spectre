// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace grmhd::ValenciaDivClean::fd {
void Reconstructor::pup([[maybe_unused]] PUP::er& p) {
#if defined(SPECTRE_USE_CHARM)
  PUP::able::pup(p);
#endif  // SPECTRE_USE_CHARM
}
}  // namespace grmhd::ValenciaDivClean::fd
