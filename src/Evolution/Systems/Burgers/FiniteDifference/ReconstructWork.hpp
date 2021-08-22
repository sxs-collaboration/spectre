// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/Tags/Inactive.hpp"
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
template <typename F>
void reconstruct_prims_work(
    gsl::not_null<std::array<
        Variables<tmpl::list<
            Tags::U, ::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>>,
        1>*>
        vars_on_lower_face,
    gsl::not_null<std::array<
        Variables<tmpl::list<
            Tags::U, ::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>>,
        1>*>
        vars_on_upper_face,
    const F& reconstruct, const Variables<tmpl::list<Tags::U>>& volume_vars,
    const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, size_t ghost_zone_size) noexcept;

/*!
 * \brief Reconstructs \f$U\f$. All results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 */
template <typename F0, typename F1>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<tmpl::list<
        Tags::U, ::Tags::Flux<Tags::U, tmpl::size_t<1>, Frame::Inertial>>>*>
        vars_on_face,
    const F0& reconstruct_lower_neighbor, const F1& reconstruct_upper_neighbor,
    const Variables<tmpl::list<Tags::U>>& subcell_volume_vars,
    const Element<1>& element,
    const FixedHashMap<maximum_number_of_neighbors(1) + 1,
                       std::pair<Direction<1>, ElementId<1>>,
                       evolution::dg::subcell::NeighborData,
                       boost::hash<std::pair<Direction<1>, ElementId<1>>>>
        neighbor_data,
    const Mesh<1>& subcell_mesh, const Direction<1>& direction_to_reconstruct,
    size_t ghost_zone_size) noexcept;
}  // namespace Burgers::fd
