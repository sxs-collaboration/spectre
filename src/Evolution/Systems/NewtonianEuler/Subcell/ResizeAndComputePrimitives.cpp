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
void ResizeAndComputePrims<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        prim_vars,
    const evolution::dg::subcell::ActiveGrid active_grid,
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, 2>& equation_of_state) {
  const size_t num_grid_points =
      (active_grid == evolution::dg::subcell::ActiveGrid::Dg ? dg_mesh
                                                             : subcell_mesh)
          .number_of_grid_points();
  if (prim_vars->number_of_grid_points() != num_grid_points) {
    prim_vars->initialize(num_grid_points);
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
#undef DIM
}  // namespace NewtonianEuler::subcell
