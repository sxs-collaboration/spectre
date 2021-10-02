// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/TciOnFdGrid.hpp"

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief The troubled-cell indicator run on the FD grid to check if the
 * corresponding DG solution is admissible.
 *
 * See `grmhd::ValenciaDivClean::subcell::TciOnFdGrid` for details.
 */
template <typename RecoveryScheme>
struct TciOnFdGrid
    : grmhd::ValenciaDivClean::subcell::TciOnFdGrid<RecoveryScheme> {};
}  // namespace grmhd::GhValenciaDivClean::subcell
