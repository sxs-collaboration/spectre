// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
template <typename TagsList>
class Variables;
namespace evolution::dg::subcell {
class NeighborData;
}  // namespace evolution::dg::subcell
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarAdvection::fd {
/*!
 * \brief Reconstructs the scalar field \f$U\f$. All results are written into
 * `vars_on_lower_face` and `vars_on_upper_face`.
 */
template <size_t Dim, typename TagsList, typename Reconstructor>
void reconstruct_work(
    gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, Dim>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<tmpl::list<Tags::U>> volume_vars,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_data,
    const Mesh<Dim>& subcell_mesh, const size_t ghost_zone_size);

/*!
 * \brief Reconstructs the scalar field \f$U\f$ for a given direction. All
 * results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct \f$U\f$ with their subcell
 * neighbors' solution (ghost data received) on the shared faces.
 */
template <size_t Dim, typename TagsList, typename ReconstructLower,
          typename ReconstructUpper>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const ReconstructLower& reconstruct_lower_neighbor,
    const ReconstructUpper& reconstruct_upper_neighbor,
    const Variables<tmpl::list<Tags::U>>& subcell_volume_vars,
    const Element<Dim>& element,
    const FixedHashMap<maximum_number_of_neighbors(Dim) + 1,
                       std::pair<Direction<Dim>, ElementId<Dim>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>
        neighbor_data,
    const Mesh<Dim>& subcell_mesh,
    const Direction<Dim>& direction_to_reconstruct,
    const size_t ghost_zone_size);
}  // namespace ScalarAdvection::fd
