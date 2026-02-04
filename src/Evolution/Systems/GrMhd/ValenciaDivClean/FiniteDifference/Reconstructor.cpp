// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace grmhd::ValenciaDivClean::fd {
void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace grmhd::ValenciaDivClean::fd
