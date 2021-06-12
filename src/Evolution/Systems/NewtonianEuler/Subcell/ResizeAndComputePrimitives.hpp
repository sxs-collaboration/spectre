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
 * \brief Mutator that resizes the primitive variables to have the size of the
 * active mesh and then computes the primitive variables on the active mesh.
 *
 * In the DG-subcell `step_actions` list this will normally be called using the
 * `::Actions::MutateApply` action right after the
 * `evolution::dg::subcell::Actions::TciAndSwitchToDg` action. We only need to
 * compute the primitives if we switched to the DG grid because otherwise we
 * computed the primitives during the FD TCI. After the primitive variables tag
 * is resized for the DG grid, the primitives are computed directly on the DG
 * grid from the reconstructed conserved variables, not via a reconstruction
 * operation applied to the primitives.
 */
template <size_t Dim>
struct ResizeAndComputePrims {
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
      tmpl::list<evolution::dg::subcell::Tags::ActiveGrid,
                 domain::Tags::Mesh<Dim>,
                 evolution::dg::subcell::Tags::Mesh<Dim>, Tags::MassDensityCons,
                 Tags::MomentumDensity<Dim>, Tags::EnergyDensity,
                 hydro::Tags::EquationOfStateBase>;

  template <size_t ThermodynamicDim>
  static void apply(
      gsl::not_null<Variables<
          tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
          prim_vars,
      evolution::dg::subcell::ActiveGrid active_grid, const Mesh<Dim>& dg_mesh,
      const Mesh<Dim>& subcell_mesh,
      const Scalar<DataVector>& mass_density_cons,
      const tnsr::I<DataVector, Dim>& momentum_density,
      const Scalar<DataVector>& energy_density,
      const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
          equation_of_state) noexcept;
};
}  // namespace NewtonianEuler::subcell
