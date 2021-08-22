// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace Burgers::subcell {
bool DgInitialDataTci::apply(
    const Variables<tmpl::list<Burgers::Tags::U>>& dg_vars,
    const Variables<tmpl::list<Inactive<Burgers::Tags::U>>>& subcell_vars,
    double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
    const Mesh<1>& dg_mesh) noexcept {
  constexpr double persson_tci_epsilon = 1.0e-18;
  return evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon) or
         evolution::dg::subcell::persson_tci(get<Burgers::Tags::U>(dg_vars),
                                             dg_mesh, persson_exponent,
                                             persson_tci_epsilon);
}
}  // namespace Burgers::subcell
