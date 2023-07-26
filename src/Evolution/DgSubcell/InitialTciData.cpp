// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/InitialTciData.hpp"

#include <pup.h>

#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace evolution::dg::subcell {
void InitialTciData::pup(PUP::er& p) {
  p | tci_status;
  p | initial_rdmp_data;
}
}  // namespace evolution::dg::subcell
