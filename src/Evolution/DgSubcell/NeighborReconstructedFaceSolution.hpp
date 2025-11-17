// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
namespace db {
class Access;
}  // namespace db
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Invoked in directions where the neighbor is doing subcell, this
 * function computes the neighbor data on the mortar via reconstruction on
 * nearest neighbor subcells.
 *
 * A list of all the directions that are doing subcell is created and then
 * passed to the mutator
 * `Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::apply`,
 * which must return a
 *
 * \code
 *  DirectionalIdMap<volume_dim, DataVector>
 * \endcode
 *
 * which holds the reconstructed `dg_packaged_data` on the face (stored in the
 * `DataVector`) for the boundary correction. A
 * `std::vector<DirectionalId<volume_dim>>`
 * holding the list of mortars that need to be reconstructed to is passed in as
 * the last argument to
 * `Metavariables::SubcellOptions::DgComputeSubcellNeighborPackagedData::apply`.
 */
template <size_t VolumeDim, typename DgComputeSubcellNeighborPackagedData>
void neighbor_reconstructed_face_solution(gsl::not_null<db::Access*> box);
}  // namespace evolution::dg::subcell
