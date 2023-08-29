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
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::subcell::detail {
template <size_t Dim>
bool persson_tci_impl(const gsl::not_null<DataVector*> filtered_component,
                      const DataVector& component, const Mesh<Dim>& dg_mesh,
                      const double alpha) {
  ASSERT(component.size() == dg_mesh.number_of_grid_points(),
         "The tensor components being checked must have the same number of "
         "grid points as the DG mesh. The tensor has "
             << component.size() << " grid points while the DG mesh has "
             << dg_mesh.number_of_grid_points() << " grid points.");

  const double component_norm{l2Norm(component)};

  const Matrix identity{};
  for (size_t d = 0; d < Dim; ++d) {
    auto matrices = make_array<Dim>(std::cref(identity));
    gsl::at(matrices, d) = Spectral::filtering::zero_lowest_modes(
        dg_mesh.slice_through(d), dg_mesh.extents(d) - 1);
    apply_matrices(filtered_component, matrices, component, dg_mesh.extents());

    //
    // Note by Yoonsoo Kim, Oct 2021 :
    //
    // The original implementation was
    //
    // ```
    // if (l2Norm(*filtered_component) <= zero_cutoff * l2Norm(component)) {
    //   continue;
    // }
    // if (std::log10(l2Norm(*filtered_component) / l2Norm(component)) >
    //     -alpha * std::log10(static_cast<double>(dg_mesh.extents(d) - 1))) {
    //   return true;
    // }
    // ```
    //
    // This form requires the extra use of `zero_cutoff` to prevent log function
    // acting on zero or a very small number, and also can be problematic if the
    // l2Norm of component is computed to be zero (if U < ~10^{-160}). The
    // evaluation of the TCI criteria is switched into a mathematically
    // equivalent but less error-prone form using power instead of log.
    //
    // Below is the benchmark result for a 3D DG element with 5 grid points per
    // dimension:
    // -------------------------------------------------------------------------
    //                                                 Non-troubled    Troubled
    //                                                   element        element
    // -------------------------------------------------------------------------
    // 1. previous implementation using log10            1262 ns         224 ns
    // 2. new implementation with pow<double,double>     1091 ns         197 ns
    // 3. using templated pow<int,int>                   1015 ns         174 ns
    // -------------------------------------------------------------------------
    //
    // Case 3 requires the Persson alpha exponent to be a compile time
    // (hard-coded) integer, which may be too restrictive. Given that switching
    // to pow<double,double> already provides a certain amount of speed-up, it
    // would be sufficient to stick to the case 2 for now.
    //
    if (pow(dg_mesh.extents(d) - 1, alpha) * l2Norm(*filtered_component) >
        component_norm) {
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
      double alpha);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::subcell::detail
