// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
/// \endcond

namespace grmhd::GhValenciaDivClean::subcell {
/*!
 * \brief Mutator that resizes the primitive variables to the subcell mesh and
 * computes the primitives, but only if
 * `evolution::dg::subcell::Tags::DidRollback` is `true`.
 *
 * In the DG-subcell `step_actions` list this will normally be called using the
 * `::Actions::MutateApply` action in the following way in the action list:
 * - `Actions::Label<Labels::BeginSubcellAfterDgRollback>`
 * - `Actions::MutateApply<PrimsAfterRollback<primitive_recovery_schemes>>`
 */
template <typename OrderedListOfRecoverySchemes>
struct PrimsAfterRollback {
  using return_tags =
      tmpl::list<::Tags::Variables<hydro::grmhd_tags<DataVector>>>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::DidRollback,
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
      bool did_rollback, const Mesh<3>& dg_mesh, const Mesh<3>& subcell_mesh,
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,
      const tnsr::aa<DataVector, 3, Frame::Inertial>& spacetime_metric,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos);
};
}  // namespace grmhd::GhValenciaDivClean::subcell
