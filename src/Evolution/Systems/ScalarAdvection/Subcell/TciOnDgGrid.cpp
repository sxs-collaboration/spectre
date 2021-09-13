// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarAdvection/Subcell/TciOnDgGrid.hpp"

#include <cstddef>

#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace ScalarAdvection::subcell {
template <size_t Dim>
bool TciOnDgGrid<Dim>::apply(const Scalar<DataVector>& dg_u,
                             const Mesh<Dim>& dg_mesh,
                             const double persson_exponent) noexcept {
  constexpr double persson_tci_epsilon = 1.0e-18;
  return ::evolution::dg::subcell::persson_tci(dg_u, dg_mesh, persson_exponent,
                                               persson_tci_epsilon);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template struct TciOnDgGrid<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef INSTANTIATION

#undef DIM
}  // namespace ScalarAdvection::subcell
