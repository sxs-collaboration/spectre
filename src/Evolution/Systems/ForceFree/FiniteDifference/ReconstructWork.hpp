// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/System.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
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
namespace gsl {
template <typename>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree::fd {
/*!
 * \brief Reconstructs the evolved variables \f$\tilde{E}^i, \tilde{B}^i,
 * \tilde{\psi}, \tilde{\phi}, \tilde{q}\f$ and the generalized electric current
 * density \f$\tilde{J}^i\f$. All results are written into `vars_on_lower_face`
 * and `vars_on_upper_face`.
 *
 */
template <typename TagsList, typename Reconstructor>
void reconstruct_work(
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const Reconstructor& reconstruct,
    const Variables<System::variables_tag::tags_list>& volume_evolved_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& volume_tilde_j,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data,
    const Mesh<3>& subcell_mesh, const size_t ghost_zone_size);

/*!
 * \brief Reconstructs the evolved variables \f$\tilde{E}^i, \tilde{B}^i,
 * \tilde{\psi}, \tilde{\phi}, \tilde{q}\f$ and the generalized electric current
 * density \f$\tilde{J}^i\f$.
 *
 * All results are written into `vars_on_face`.
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
    const Variables<System::variables_tag::tags_list>&
        subcell_volume_evolved_vars,
    const tnsr::I<DataVector, 3, Frame::Inertial>& subcell_volume_tilde_j,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size);

}  // namespace ForceFree::fd
