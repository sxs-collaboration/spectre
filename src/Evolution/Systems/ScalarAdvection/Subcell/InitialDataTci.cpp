// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/InitialDataTci.hpp"

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Evolution/DgSubcell/TwoMeshRdmpTci.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
bool DgInitialDataTci<Dim>::apply(
    const Variables<tmpl::list<ScalarAdvection::Tags::U>>& dg_vars,
    const Variables<tmpl::list<Inactive<ScalarAdvection::Tags::U>>>&
        subcell_vars,
    const Mesh<Dim>& dg_mesh, const double persson_exponent,
    const double rdmp_delta0, const double rdmp_epsilon) {
  constexpr double persson_tci_epsilon = 1.0e-18;
  return evolution::dg::subcell::persson_tci(
             get<ScalarAdvection::Tags::U>(dg_vars), dg_mesh, persson_exponent,
             persson_tci_epsilon) or
         evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct DgInitialDataTci<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
