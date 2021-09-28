// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/DidRollback.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
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

namespace NewtonianEuler::subcell {
/*!
 * \brief Mutator that resizes the primitive variables to the subcell mesh and
 * computes the primitives, but only if
 * `evolution::dg::subcell::Tags::DidRollback` is `true`.
 *
 * In the DG-subcell `step_actions` list this will normally be called using the
 * `::Actions::MutateApply` action right after the
 * `evolution::dg::subcell::Actions::Labels::BeginSubcellAfterDgRollback` label.
 */
template <size_t Dim>
struct PrimsAfterRollback {
 private:
  using MassDensity = NewtonianEuler::Tags::MassDensity<DataVector>;
  using Velocity = NewtonianEuler::Tags::Velocity<DataVector, Dim>;
  using SpecificInternalEnergy =
      NewtonianEuler::Tags::SpecificInternalEnergy<DataVector>;
  using Pressure = NewtonianEuler::Tags::Pressure<DataVector>;

 public:
  using return_tags = tmpl::list<::Tags::Variables<
      tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>>;
  using argument_tags =
      tmpl::list<evolution::dg::subcell::Tags::DidRollback,
                 evolution::dg::subcell::Tags::Mesh<Dim>, Tags::MassDensityCons,
                 Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
                 hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
          prim_vars,
      bool did_rollback, const Mesh<Dim>& subcell_mesh,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state);
};
}  // namespace NewtonianEuler::subcell
