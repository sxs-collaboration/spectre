// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Burgers/Subcell/TciOnFdGrid.hpp"

#include "Evolution/DgSubcell/PerssonTci.hpp"

namespace Burgers::subcell {
bool TciOnFdGrid::apply(const Scalar<DataVector>& dg_u, const Mesh<1>& dg_mesh,
                        const double persson_exponent) {
  return ::evolution::dg::subcell::persson_tci(dg_u, dg_mesh, persson_exponent);
}
}  // namespace Burgers::subcell
