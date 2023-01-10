// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/Tags/TciStatus.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell::Tags {
template <size_t Dim>
void TciStatusCompute<Dim>::function(const gsl::not_null<return_type*> result,
                                     const int tci_decision,
                                     const subcell::ActiveGrid active_grid,
                                     const ::Mesh<Dim>& subcell_mesh,
                                     const ::Mesh<Dim>& dg_mesh) {
  get(*result).destructive_resize(active_grid == subcell::ActiveGrid::Dg
                                      ? dg_mesh.number_of_grid_points()
                                      : subcell_mesh.number_of_grid_points());
  get(*result) = tci_decision;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class TciStatusCompute<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::Tags
