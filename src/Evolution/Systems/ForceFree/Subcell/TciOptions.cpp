// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ForceFree/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace ForceFree::subcell {
void TciOptions::pup(PUP::er& p) { p | tilde_q_cutoff; }
}  // namespace ForceFree::subcell
