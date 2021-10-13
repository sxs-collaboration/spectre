// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename TagsList>
class Variables;
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
template <size_t Dim>
class Direction;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
namespace evolution::dg::subcell {
class NeighborData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace Burgers::fd {
/*!
 * \brief Reconstructs \f$U\f$. All results are written into
 * `vars_on_lower_face` and `vars_on_upper_face`.
 */
template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    gsl::not_null<std::array<Variables<TagsList>, 1>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, 1>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<tmpl::list<Tags::U>> volume_vars, const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, const size_t ghost_zone_size);

/*!
 * \brief Reconstructs \f$U\f$. All results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 */
template <typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<tmpl::list<Tags::U>>& subcell_volume_vars,
    const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, const Direction<1>& direction_to_reconstruct,
    const size_t ghost_zone_size);
}  // namespace Burgers::fd
