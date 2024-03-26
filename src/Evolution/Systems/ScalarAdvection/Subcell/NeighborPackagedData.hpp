// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

/// \cond
namespace db {
class Access;
}  // namespace db
class DataVector;
template <size_t VolumeDim>
struct DirectionalId;
template <size_t Dim, typename T>
class DirectionalIdMap;
/// \endcond

namespace ScalarAdvection::subcell {
/*!
 * \brief On elements using DG, reconstructs the interface data from a
 * neighboring element doing subcell.
 *
 * The neighbor's packaged data needed by the boundary correction is computed
 * and returned so that it can be used for solving the Riemann problem on the
 * interfaces.
 *
 * Note that for strict conservation the Riemann solve should be done on the
 * subcells, with the correction being projected back to the DG interface.
 * However, in practice such strict conservation doesn't seem to be necessary
 * and can be explained by that we only need strict conservation at shocks, and
 * if one element is doing DG, then we aren't at a shock.
 */

struct NeighborPackagedData {
  template <size_t Dim>
  static DirectionalIdMap<Dim, DataVector> apply(
      const db::Access& box,
      const std::vector<DirectionalId<Dim>>& mortars_to_reconstruct_to);
};
}  // namespace ScalarAdvection::subcell
