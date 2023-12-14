// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/GrMhd/GhValenciaDivClean/Tags.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/ReconstructWork.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

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
namespace evolution::dg::Actions::detail {
template <size_t Dim>
struct NormalVector;
}  // namespace evolution::dg::Actions::detail
namespace gh::ConstraintDamping::Tags {
struct ConstraintGamma1;
struct ConstraintGamma2;
}  // namespace gh::ConstraintDamping::Tags
/// \endcond

namespace grmhd::GhValenciaDivClean::fd {
using tags_list_for_reconstruct =
    tmpl::push_front<grmhd::ValenciaDivClean::fd::tags_list_for_reconstruct,
                     gr::Tags::SpacetimeMetric<DataVector, 3>,
                     gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>;

namespace detail {
using tags_list_for_reconstruct_split_lapse =
    tmpl::split<tags_list_for_reconstruct, gr::Tags::Lapse<DataVector>>;
}  // namespace detail

using tags_list_for_reconstruct_fd_neighbor = tmpl::append<
    tmpl::front<detail::tags_list_for_reconstruct_split_lapse>,
    tmpl::push_front<tmpl::back<detail::tags_list_for_reconstruct_split_lapse>,
                     gh::ConstraintDamping::Tags::ConstraintGamma1,
                     gh::ConstraintDamping::Tags::ConstraintGamma2,
                     gr::Tags::Lapse<DataVector>>>;

/*!
 * \brief Reconstructs \f$\rho, p, Wv^i, B^i\f$, \f$\Phi\f$, and the spacetime
 * metric, then computes the Lorentz factor, upper spatial velocity, specific
 * internal energy, and the conserved variables. All results are written into
 * `vars_on_lower_face` and `vars_on_upper_face`.
 */
template <typename SpacetimeTagsToReconstruct,
          typename PrimTagsForReconstruction, typename PrimsTags,
          typename SpacetimeAndConsTags, size_t ThermodynamicDim,
          typename HydroReconstructor, typename SpacetimeReconstructor,
          typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags,
          typename PrimsTagsSentByNeighbor>
void reconstruct_prims_work(
    gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_lower_face,
    gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_upper_face,
    const HydroReconstructor& hydro_reconstructor,
    const SpacetimeReconstructor& spacetime_reconstructor,
    const ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags&
        spacetime_vars_for_grmhd,
    const Variables<PrimsTags>& volume_prims,
    const Variables<SpacetimeAndConsTags>& volume_spacetime_and_cons_vars,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, Variables<PrimsTagsSentByNeighbor>>&
        neighbor_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size,
    bool compute_conservatives);

/*!
 * \brief Reconstructs \f$\rho, p, Wv^i, B^i\f$, \f$\Phi\f$, the spacetime
 * metric, \f$\Phi_{iab}\f$, and \f$\Pi_{ab}\f$, then computes the Lorentz
 * factor, upper spatial velocity, specific internal energy, and the conserved
 * variables. All results are written into `vars_on_face`.
 *
 * This is used on DG elements to reconstruct their subcell neighbors' solution
 * on the shared faces.
 */
template <
    typename SpacetimeTagsToReconstruct, typename PrimTagsForReconstruction,
    typename PrimsTagsSentByNeighbor, typename PrimsTags,
    size_t ThermodynamicDim, typename LowerHydroReconstructor,
    typename LowerSpacetimeReconstructor, typename UpperHydroReconstructor,
    typename UpperSpacetimeReconstructor,
    typename ComputeGrmhdSpacetimeVarsFromReconstructedSpacetimeTags>
void reconstruct_fd_neighbor_work(
    gsl::not_null<Variables<tags_list_for_reconstruct_fd_neighbor>*>
        vars_on_face,
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
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    size_t ghost_zone_size, bool compute_conservatives);
}  // namespace grmhd::GhValenciaDivClean::fd
