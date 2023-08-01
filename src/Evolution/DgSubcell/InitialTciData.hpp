// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>

#include "Evolution/DgSubcell/RdmpTciData.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Used to communicate the RDMP and TCI status/decision during
 * initialization.
 */
struct InitialTciData {
  std::optional<int> tci_status{};
  std::optional<evolution::dg::subcell::RdmpTciData> initial_rdmp_data{};

  void pup(PUP::er& p);
};
}  // namespace evolution::dg::subcell
