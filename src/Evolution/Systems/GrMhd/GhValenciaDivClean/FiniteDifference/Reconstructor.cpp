// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace grmhd::GhValenciaDivClean::fd {
Reconstructor::Reconstructor(CkMigrateMessage* const msg) : PUP::able(msg) {}

void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace grmhd::GhValenciaDivClean::fd
