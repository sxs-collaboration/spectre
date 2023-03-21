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
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"

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

namespace grmhd::GhValenciaDivClean::fd {
/*!
 * \brief Reconstructs \f$\rho, p, Wv^i, B^i\f$, \f$\Phi\f$, and the spacetime
 * metric, then computes the Lorentz factor, upper spatial velocity, specific
 * internal energy, specific enthalpy, and the conserved variables. All results
 * are written into `vars_on_lower_face` and `vars_on_upper_face`.
 */
template <typename SpacetimeTagsToReconstruct,
          typename PrimTagsForReconstruction, typename PrimsTags,
          typename SpacetimeAndConsTags, typename TagsList,
          size_t ThermodynamicDim, typename HydroReconstructor,
          typename SpacetimeReconstructor,
          typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags,
          typename PrimsTagsSentByNeighbor>
void reconstruct_prims_work(
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_lower_face,
    gsl::not_null<std::array<Variables<TagsList>, 3>*> vars_on_upper_face,
    const HydroReconstructor& hydro_reconstructor,
    const SpacetimeReconstructor& spacetime_reconstructor,
    const ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags&
        spacetime_vars_for_grmhd,
    const Variables<PrimsTags>& volume_prims,
    const Variables<SpacetimeAndConsTags>& volume_spacetime_and_cons_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        Variables<PrimsTagsSentByNeighbor>,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& neighbor_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size,
    bool compute_conservatives);

/*!
 * \brief Reconstructs \f$\rho, p, Wv^i, B^i\f$, \f$\Phi\f$, the spacetime
 * metric, \f$\Phi_{iab}\f$, and \f$\Pi_{ab}\f$, then computes the Lorentz
 * factor, upper spatial velocity, specific internal energy, specific enthalpy,
 * and the conserved variables. All results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 */
template <
    typename SpacetimeTagsToReconstruct, typename PrimTagsForReconstruction,
    typename PrimsTagsSentByNeighbor, typename TagsList, typename PrimsTags,
    size_t ThermodynamicDim, typename LowerHydroReconstructor,
    typename LowerSpacetimeReconstructor, typename UpperHydroReconstructor,
    typename UpperSpacetimeReconstructor,
    typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<TagsList>*> vars_on_face,
    const LowerHydroReconstructor& reconstruct_lower_neighbor_hydro,
    const LowerSpacetimeReconstructor& reconstruct_lower_neighbor_spacetime,
    const UpperHydroReconstructor& reconstruct_upper_neighbor_hydro,
    const UpperSpacetimeReconstructor& reconstruct_upper_neighbor_spacetime,
    const ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags&
        spacetime_vars_for_grmhd,
    const Variables<PrimsTags>& subcell_volume_prims,
    const Variables<
        grmhd::GhValenciaDivClean::Tags::spacetime_reconstruction_tags>&
        subcell_volume_spacetime_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const FixedHashMap<
        maximum_number_of_neighbors(3), std::pair<Direction<3>, ElementId<3>>,
        evolution::dg::subcell::GhostData,
        boost::hash<std::pair<Direction<3>, ElementId<3>>>>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    size_t ghost_zone_size, bool compute_conservatives);
}  // namespace grmhd::GhValenciaDivClean::fd
