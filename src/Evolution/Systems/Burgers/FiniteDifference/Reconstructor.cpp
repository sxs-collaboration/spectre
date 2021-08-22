// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace Burgers::fd {
Reconstructor::Reconstructor(CkMigrateMessage* const msg) noexcept
    : PUP::able(msg) {}

void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace Burgers::fd
