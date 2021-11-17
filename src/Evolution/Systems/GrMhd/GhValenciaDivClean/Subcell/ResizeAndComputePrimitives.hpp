// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
template <size_t Dim>
class Mesh;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
template <typename TagsList>
class Variables;
/// \endcond

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief If the grid is switched from subcell to DG, then this mutator resizes
 * the primitive variables to the DG grid and computes the primitive variables
 * on the DG grid.
 *
 * In the DG-subcell `step_actions` list this will normally be called using the
 * `::Actions::MutateApply` action in the following way in the action list:
 * - `TciAndSwitchToDg<TciOnFdGrid>`
 * - `Actions::MutateApply<ResizeAndComputePrims<primitive_recovery_schemes>>`
 *
 * If the active grid is DG (we are switching from subcell back to DG) then this
 * mutator computes the primitive variables on the active grid. We reconstruct
 * the pressure to the DG grid to give a high-order initial guess for the
 * primitive recovery. A possible future optimization would be to avoid this
 * reconstruction when all recovery schemes don't need an initial guess.
 * Finally, we perform the primitive recovery on the DG grid.
 *
 * If the active grid is Subcell then this mutator does nothing.
 *
 * \note All evolved variables are on the DG grid when this mutator is called
 * and the active grid is DG.
 */
template <typename OrderedListOfRecoverySchemes>
struct ResizeAndComputePrims {
  using return_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::ActiveGrid,
                 domain::Tags::Mesh<3>, evolution::dg::subcell::Tags::Mesh<3>,
                 grmhd::ValenciaDivClean::Tags::TildeD,
                 grmhd::ValenciaDivClean::Tags::TildeTau,
                 grmhd::ValenciaDivClean::Tags::TildeS<>,
                 grmhd::ValenciaDivClean::Tags::TildeB<>,
                 grmhd::ValenciaDivClean::Tags::TildePhi,
                 gr::Tags::SpacetimeMetric<3>,
                 hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Variables<hydro::grmhd_tags<DataVector>>*> prim_vars,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<3>& dg_mesh,
      const Mesh<3>& subcell_mesh, const Scalar<DataVector>& tilde_d,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos);
};
}  // namespace grmhd::GhValenciaDivClean::subcell
