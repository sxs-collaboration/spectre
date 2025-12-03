// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
template <size_t VolumeDim>
class DirectionalId;
namespace db {
class Access;
}  // namespace db
namespace evolution::dg {
template <size_t Dim>
struct BoundaryData;
}  // namespace evolution::dg
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Copies the neighbors' TCI decisions into
 * `subcell::Tags::NeighborTciDecisions<Dim>`
 */
template <size_t Dim>
void neighbor_tci_decision(
    gsl::not_null<db::Access*> box,
    const DirectionalId<Dim>& directional_element_id,
    const evolution::dg::BoundaryData<Dim>& neighbor_data);
}  // namespace evolution::dg::subcell
