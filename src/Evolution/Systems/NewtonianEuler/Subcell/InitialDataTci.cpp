// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianEuler::subcell {
template <size_t Dim>
bool DgInitialDataTci<Dim>::apply(
    const Variables<
        tmpl::list<MassDensityCons, MomentumDensity, EnergyDensity>>& dg_vars,
    const Variables<
        tmpl::list<Inactive<MassDensityCons>, Inactive<MomentumDensity>,
                   Inactive<EnergyDensity>>>& subcell_vars,
    const double rdmp_delta0, const double rdmp_epsilon,
    const double persson_exponent, const Mesh<Dim>& dg_mesh,
    const Scalar<DataVector>& dg_pressure) noexcept {
  return evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon) or
         evolution::dg::subcell::persson_tci(get<MassDensityCons>(dg_vars),
                                             dg_mesh, persson_exponent,
                                             1.0e-18) or
         evolution::dg::subcell::persson_tci(dg_pressure, dg_mesh,
                                             persson_exponent, 1.0e-18);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATION(r, data) template struct DgInitialDataTci<DIM(data)>;
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))
#undef INSTANTIATION
#undef DIM
}  // namespace NewtonianEuler::subcell
