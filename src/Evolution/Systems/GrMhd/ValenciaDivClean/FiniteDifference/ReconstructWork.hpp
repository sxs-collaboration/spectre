// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"

/// \cond
class DataVector;
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
namespace EquationsOfState {
template <bool IsRelativistic, size_t ThermodynamicDim>
class EquationOfState;
}  // namespace EquationsOfState
namespace evolution::dg::subcell {
class GhostData;
}  // namespace evolution::dg::subcell
/// \endcond

namespace grmhd::ValenciaDivClean::fd {
/*!
 * \brief Reconstructs the `PrimTagsForReconstruction` (usually
 * \f$\rho, p, Wv^i, B^i\f$, and \f$\Phi\f$), and if `compute_conservatives` is
 * true computes the Lorentz factor, upper spatial velocity, specific internal
 * energy, specific enthalpy, and the conserved variables.
 *
 * All results are written into `vars_on_lower_face` and `vars_on_upper_face`.
 *
 * The reason the `PrimTagsForReconstruction` can be specified separately is
 * because some variables might need separate reconstruction methods from
 * others, e.g. to guarantee the reconstructed solution is positive.
 */
template <typename PrimTagsForReconstruction, typename PrimsTagsVolume,
          typename TagsList, size_t ThermodynamicDim, typename F,
          typename PrimsTagsSentByNeighbor>
void reconstruct_prims_work(
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const F& reconstruct, const Variables<PrimsTagsVolume>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        Variables<PrimsTagsSentByNeighbor>,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size,
    bool compute_conservatives);

/*!
 * \brief Reconstructs the `PrimTagsForReconstruction` and if
 * `compute_conservatives` is `true`  computes the Lorentz factor, upper spatial
 * velocity, specific internal energy, specific enthalpy, and the conserved
 * variables.
 *
 * All results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 *
 * The reason the `PrimTagsForReconstruction` can be specified separately is
 * because some variables might need separate reconstruction methods from
 * others, e.g. to guarantee the reconstructed solution is positiv
 */
template <typename PrimTagsForReconstruction, typename PrimsTagsSentByNeighbor,
          typename TagsList, typename PrimsTags, size_t ThermodynamicDim,
          typename F0, typename F1>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const F0& reconstruct_lower_neighbor, const F1& reconstruct_upper_neighbor,
    const Variables<PrimsTags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size, bool compute_conservatives);
}  // namespace grmhd::ValenciaDivClean::fd
