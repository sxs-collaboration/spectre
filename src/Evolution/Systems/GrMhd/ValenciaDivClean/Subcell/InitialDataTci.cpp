// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace grmhd::ValenciaDivClean::subcell {
bool DgInitialDataTci::apply(
    const Variables<tmpl::list<
        ValenciaDivClean::Tags::TildeD, ValenciaDivClean::Tags::TildeTau,
        ValenciaDivClean::Tags::TildeS<>, ValenciaDivClean::Tags::TildeB<>,
        ValenciaDivClean::Tags::TildePhi>>& dg_vars,
    const Variables<tmpl::list<Inactive<ValenciaDivClean::Tags::TildeD>,
                               Inactive<ValenciaDivClean::Tags::TildeTau>,
                               Inactive<ValenciaDivClean::Tags::TildeS<>>,
                               Inactive<ValenciaDivClean::Tags::TildeB<>>,
                               Inactive<ValenciaDivClean::Tags::TildePhi>>>&
        subcell_vars,
    double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
    const Mesh<3>& dg_mesh) noexcept {
  return evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon) or
         evolution::dg::subcell::persson_tci(
             get<Tags::TildeD>(dg_vars), dg_mesh, persson_exponent, 1.0e-18) or
         evolution::dg::subcell::persson_tci(
             get<Tags::TildeTau>(dg_vars), dg_mesh, persson_exponent, 1.0e-18);
}
}  // namespace grmhd::ValenciaDivClean::subcell
