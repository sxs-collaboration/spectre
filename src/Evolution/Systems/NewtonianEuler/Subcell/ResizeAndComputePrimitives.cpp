// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/ResizeAndComputePrimitives.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
template <size_t ThermodynamicDim>
void ResizeAndComputePrims<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        prim_vars,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points =
      (active_grid == evolution::dg::subcell::ActiveGrid::Dg ? dg_mesh
                                                             : subcell_mesh)
          .number_of_grid_points();
  if (prim_vars->number_of_grid_points() != num_grid_points) {
    ASSERT(active_grid == evolution::dg::subcell::ActiveGrid::Dg,
           "ResizeAndComputePrims should only be resizing when switching from "
           "subcell to DG");
    prim_vars->initialize(num_grid_points);

    // We only need to compute the prims if we switched to the DG grid because
    // otherwise we computed the prims during the FD TCI.
    NewtonianEuler::PrimitiveFromConservative<Dim>::apply(
        make_not_null(&get<MassDensity>(*prim_vars)),
        make_not_null(&get<Velocity>(*prim_vars)),
        make_not_null(&get<SpecificInternalEnergy>(*prim_vars)),
        make_not_null(&get<Pressure>(*prim_vars)), mass_density_cons,
        momentum_density, energy_density, equation_of_state);
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class ResizeAndComputePrims<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                \
  template void ResizeAndComputePrims<DIM(data)>::apply<THERMO_DIM(data)>(    \
      gsl::not_null<Variables<tmpl::list<MassDensity, Velocity,               \
                                         SpecificInternalEnergy, Pressure>>*> \
          prim_vars,                                                          \
      evolution::dg::subcell::ActiveGrid active_grid,                         \
      const Mesh<DIM(data)>& dg_mesh, const Mesh<DIM(data)>& subcell_mesh,    \
      const Scalar<DataVector>& mass_density_cons,                            \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                 \
      const Scalar<DataVector>& energy_density,                               \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&       \
          equation_of_state);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::subcell
