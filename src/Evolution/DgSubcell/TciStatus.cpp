// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/TciStatus.hpp"

#include <cstddef>
#include <deque>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/ActiveGrid.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace evolution::dg::subcell {
template <size_t Dim>
void tci_status(const gsl::not_null<Scalar<DataVector>*> status,
                const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
                const subcell::ActiveGrid active_grid,
                const std::deque<subcell::ActiveGrid>& tci_history) {
  const auto set_status =
      [&active_grid, &dg_mesh, &status,
       &subcell_mesh](const subcell::ActiveGrid grid_to_set_from) {
        destructive_resize_components(
            status, active_grid == ActiveGrid::Dg
                        ? dg_mesh.number_of_grid_points()
                        : subcell_mesh.number_of_grid_points());
        if (grid_to_set_from == ActiveGrid::Dg) {
          get(*status) = 0.0;
        } else {
          get(*status) = 1.0;
        }
      };
  if (tci_history.empty()) {
    set_status(active_grid);
  } else {
    set_status(tci_history.front());
  }
}

template <size_t Dim>
Scalar<DataVector> tci_status(
    const Mesh<Dim>& dg_mesh, const Mesh<Dim>& subcell_mesh,
    const subcell::ActiveGrid active_grid,
    const std::deque<subcell::ActiveGrid>& tci_history) {
  Scalar<DataVector> status{};
  tci_status(make_not_null(&status), dg_mesh, subcell_mesh, active_grid,
             tci_history);
  return status;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template void tci_status(                                                \
      gsl::not_null<Scalar<DataVector>*> status,                           \
      const Mesh<DIM(data)>& dg_mesh, const Mesh<DIM(data)>& subcell_mesh, \
      subcell::ActiveGrid active_grid,                                     \
      const std::deque<subcell::ActiveGrid>& tci_history);                 \
  template Scalar<DataVector> tci_status(                                  \
      const Mesh<DIM(data)>& dg_mesh, const Mesh<DIM(data)>& subcell_mesh, \
      subcell::ActiveGrid active_grid,                                     \
      const std::deque<subcell::ActiveGrid>& tci_history);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell
