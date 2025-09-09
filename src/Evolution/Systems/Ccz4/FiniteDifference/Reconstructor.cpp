// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Ccz4/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace Ccz4::fd {
Reconstructor::Reconstructor(CkMigrateMessage* const msg) : PUP::able(msg) {}

void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace Ccz4::fd
