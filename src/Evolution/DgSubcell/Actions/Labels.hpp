// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace evolution::dg::subcell::Actions {
/// Labels used to navigate the action list when using a DG-subcell scheme
namespace Labels {
/// Label marking the start of the unlimited DG solver
struct BeginDg {};
/// Label marking the end of the `step_actions`, i.e. the end of both the
/// unlimited DG solver and the subcell solver.
struct EndOfSolvers {};
/// Label marking the start of the subcell solver
struct BeginSubcell {};
/// Label marking the part of the subcell solver that the unlimited DG solver
/// jumps to after rolling back the unlimited DG step because it was
/// inadmissible.
struct BeginSubcellAfterDgRollback {};
}  // namespace Labels
}  // namespace evolution::dg::subcell::Actions
