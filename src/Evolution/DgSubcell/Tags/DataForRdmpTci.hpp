// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"

namespace evolution::dg::subcell::Tags {
/// The data for the RDMP troubled-cell indicator.
struct DataForRdmpTci : db::SimpleTag {
  using type = RdmpTciData;
};
}  // namespace evolution::dg::subcell::Tags
