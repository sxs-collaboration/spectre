// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/TciOnDgGrid.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
template <size_t ThermodynamicDim>
bool TciOnDgGrid<Dim>::apply(
    const gsl::not_null<Variables<
        tmpl::list<MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>
        dg_prim_vars,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>& eos,
    const Mesh<Dim>& dg_mesh, const double persson_exponent) noexcept {
  NewtonianEuler::PrimitiveFromConservative<Dim, ThermodynamicDim>::apply(
      make_not_null(&get<MassDensity>(*dg_prim_vars)),
      make_not_null(&get<Velocity>(*dg_prim_vars)),
      make_not_null(&get<SpecificInternalEnergy>(*dg_prim_vars)),
      make_not_null(&get<Pressure>(*dg_prim_vars)), mass_density,
      momentum_density, energy_density, eos);
  constexpr double persson_tci_epsilon = 1.0e-18;
  return min(get(mass_density)) < min_density_allowed or
         min(get(get<Pressure>(*dg_prim_vars))) < min_pressure_allowed or
         evolution::dg::subcell::persson_tci(
             mass_density, dg_mesh, persson_exponent, persson_tci_epsilon) or
         evolution::dg::subcell::persson_tci(get<Pressure>(*dg_prim_vars),
                                             dg_mesh, persson_exponent,
                                             persson_tci_epsilon);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template class TciOnDgGrid<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                                 \
  template bool TciOnDgGrid<DIM(data)>::apply<THERMO_DIM(data)>(               \
      const gsl::not_null<Variables<tmpl::list<                                \
          MassDensity, Velocity, SpecificInternalEnergy, Pressure>>*>          \
          dg_prim_vars,                                                        \
      const Scalar<DataVector>& mass_density,                                  \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& momentum_density, \
      const Scalar<DataVector>& energy_density,                                \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& eos,   \
      const Mesh<DIM(data)>& dg_mesh, const double persson_exponent) noexcept;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))
#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
}  // namespace NewtonianEuler::subcell
