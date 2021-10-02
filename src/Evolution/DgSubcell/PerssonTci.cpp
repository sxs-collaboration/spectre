// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/PerssonTci.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::detail {
template <size_t Dim>
bool persson_tci_impl(const gsl::not_null<DataVector*> filtered_component,
                      const DataVector& component, const Mesh<Dim>& dg_mesh,
                      const double alpha, const double zero_cutoff) {
  ASSERT(component.size() == dg_mesh.number_of_grid_points(),
         "The tensor components being checked must have the same number of "
         "grid points as the DG mesh. The tensor has "
             << component.size() << " grid points while the DG mesh has "
             << dg_mesh.number_of_grid_points() << " grid points.");

  const Matrix identity{};
  for (size_t d = 0; d < Dim; ++d) {
    auto matrices = make_array<Dim>(std::cref(identity));
    gsl::at(matrices, d) = Spectral::filtering::zero_lowest_modes(
        dg_mesh.slice_through(d), dg_mesh.extents(d) - 1);
    apply_matrices(filtered_component, matrices, component, dg_mesh.extents());

    // Avoid taking logs of small numbers (or worse, zero)
    if (l2Norm(*filtered_component) <= zero_cutoff * l2Norm(component)) {
      continue;
    }

    if (std::log10(l2Norm(*filtered_component) / l2Norm(component)) >
        -alpha * std::log10(static_cast<double>(dg_mesh.extents(d) - 1))) {
      return true;
    }
  }
  return false;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                     \
  template bool persson_tci_impl(                                  \
      gsl::not_null<DataVector*> filtered_component,               \
      const DataVector& component, const Mesh<DIM(data)>& dg_mesh, \
      double alpha, double zero_cutoff);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::detail
