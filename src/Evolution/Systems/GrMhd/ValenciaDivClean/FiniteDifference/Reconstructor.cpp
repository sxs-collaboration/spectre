// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace grmhd::ValenciaDivClean::fd {
Reconstructor::Reconstructor(CkMigrateMessage* const msg) : PUP::able(msg) {}

void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace grmhd::ValenciaDivClean::fd
