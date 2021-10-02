// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnDgGrid.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief The troubled-cell indicator run on the DG grid to check if the
 * solution is admissible.
 *
 * See `grmhd::ValenciaDivClean::subcell::TciOnDgGrid` for details.
 */
template <typename RecoveryScheme>
struct TciOnDgGrid
    : grmhd::ValenciaDivClean::subcell::TciOnDgGrid<RecoveryScheme> {};
}  // namespace grmhd::GhValenciaDivClean::subcell
