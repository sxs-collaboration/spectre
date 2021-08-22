// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/TciOnFdGrid.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DgSubcell/PerssonTci.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/TMPL.hpp"

namespace Burgers::subcell {
bool TciOnFdGrid::apply(const Scalar<DataVector>& dg_u, const Mesh<1>& dg_mesh,
                        const double persson_exponent) noexcept {
  constexpr double persson_tci_epsilon = 1.0e-18;
  return evolution::dg::subcell::persson_tci(dg_u, dg_mesh, persson_exponent,
                                             persson_tci_epsilon);
}
}  // namespace Burgers::subcell
