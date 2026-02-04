// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"

#include <pup.h>

namespace ForceFree::fd {
void Reconstructor::pup(PUP::er& p) { PUP::able::pup(p); }
}  // namespace ForceFree::fd
