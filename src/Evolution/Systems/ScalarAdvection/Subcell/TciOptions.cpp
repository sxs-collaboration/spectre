// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/TciOptions.hpp"

#include <pup.h>

#include "Parallel/PupStlCpp17.hpp"

namespace ScalarAdvection::subcell {
void TciOptions::pup(PUP::er& /*p*/) noexcept {}
}  // namespace ScalarAdvection::subcell
