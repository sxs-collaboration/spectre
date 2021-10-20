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
    double rdmp_delta0, double rdmp_epsilon, double persson_exponent,
    const Mesh<Dim>& dg_mesh) {
  return evolution::dg::subcell::persson_tci(
             get<ScalarAdvection::Tags::U>(dg_vars), dg_mesh,
             persson_exponent) or
         evolution::dg::subcell::two_mesh_rdmp_tci(dg_vars, subcell_vars,
                                                   rdmp_delta0, rdmp_epsilon);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct DgInitialDataTci<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
