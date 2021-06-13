// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/PrimsAfterRollback.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
template <size_t ThermodynamicDim>
void PrimsAfterRollback<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        prim_vars,
    const bool did_rollback, const Mesh<Dim>& subcell_mesh,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  if (did_rollback) {
    const size_t num_grid_points = subcell_mesh.number_of_grid_points();
    if (prim_vars->number_of_grid_points() != num_grid_points) {
      prim_vars->initialize(num_grid_points);
    }
    NewtonianEuler::PrimitiveFromConservative<Dim, ThermodynamicDim>::apply(
        make_not_null(&get<MassDensity>(*prim_vars)),
        make_not_null(&get<Velocity>(*prim_vars)),
        make_not_null(&get<SpecificInternalEnergy>(*prim_vars)),
        make_not_null(&get<Pressure>(*prim_vars)), mass_density_cons,
        momentum_density, energy_density, equation_of_state);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class PrimsAfterRollback<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template void PrimsAfterRollback<DIM(data)>::apply<THERMO_DIM(data)>(       \
      gsl::not_null<Variables<tmpl::list<MassDensity, Velocity,               \
                                         SpecificInternalEnergy, Pressure>>*> \
          prim_vars,                                                          \
      bool did_rollback, const Mesh<DIM(data)>& subcell_mesh,                 \
      const Scalar<DataVector>& mass_density_cons,                            \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                 \
      const Scalar<DataVector>& energy_density,                               \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&       \
          equation_of_state) noexcept;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::subcell
