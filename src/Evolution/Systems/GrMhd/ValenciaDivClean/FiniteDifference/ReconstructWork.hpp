// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
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
/// \endcond

namespace grmhd::ValenciaDivClean::fd {
/// \brief The list of tags used by `reconstruct_prims_work` and
/// `reconstruct_fd_neighbor_work`
using tags_list_for_reconstruct = tmpl::list<
    grmhd::ValenciaDivClean::Tags::TildeD,
    grmhd::ValenciaDivClean::Tags::TildeYe,
    grmhd::ValenciaDivClean::Tags::TildeTau,
    grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
    grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
    grmhd::ValenciaDivClean::Tags::TildePhi,
    hydro::Tags::RestMassDensity<DataVector>,
    hydro::Tags::ElectronFraction<DataVector>,
    hydro::Tags::SpecificInternalEnergy<DataVector>,
    hydro::Tags::SpatialVelocity<DataVector, 3>,
    hydro::Tags::MagneticField<DataVector, 3>,
    hydro::Tags::DivergenceCleaningField<DataVector>,
    hydro::Tags::LorentzFactor<DataVector>, hydro::Tags::Pressure<DataVector>,
    hydro::Tags::Temperature<DataVector>,
    hydro::Tags::LorentzFactorTimesSpatialVelocity<DataVector, 3>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeD, tmpl::size_t<3>,
                 Frame::Inertial>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeYe, tmpl::size_t<3>,
                 Frame::Inertial>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeTau, tmpl::size_t<3>,
                 Frame::Inertial>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>,
                 tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>,
                 tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::Flux<grmhd::ValenciaDivClean::Tags::TildePhi, tmpl::size_t<3>,
                 Frame::Inertial>,
    gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
    hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
    gr::Tags::SpatialMetric<DataVector, 3>,
    gr::Tags::SqrtDetSpatialMetric<DataVector>,
    gr::Tags::InverseSpatialMetric<DataVector, 3>,
    evolution::dg::Actions::detail::NormalVector<3>>;

/*!
 * \brief Reconstructs the `PrimTagsForReconstruction` (usually
 * \f$\rho, p, Wv^i, B^i\f$, and \f$\Phi\f$), and if `compute_conservatives` is
 * true computes the Lorentz factor, upper spatial velocity, specific internal
 * energy, and the conserved variables.
 *
 * All results are written into `vars_on_lower_face` and `vars_on_upper_face`.
 *
 * The reason the `PrimTagsForReconstruction` can be specified separately is
 * because some variables might need separate reconstruction methods from
 * others, e.g. to guarantee the reconstructed solution is positive.
 */
template <typename PrimTagsForReconstruction, typename PrimsTagsVolume,
          size_t ThermodynamicDim, typename F, typename PrimsTagsSentByNeighbor>
void reconstruct_prims_work(
    gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_lower_face,
    gsl::not_null<std::array<Variables<tags_list_for_reconstruct>, 3>*>
        vars_on_upper_face,
    const F& reconstruct, const Variables<PrimsTagsVolume>& volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, Variables<PrimsTagsSentByNeighbor>>&
        neighbor_data,
    const Mesh<3>& subcell_mesh, size_t ghost_zone_size,
    bool compute_conservatives);

/*!
 * \brief Reconstructs the `PrimTagsForReconstruction` and if
 * `compute_conservatives` is `true`  computes the Lorentz factor, upper spatial
 * velocity, specific internal energy, and the conserved variables.
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
    gsl::not_null<Variables<tags_list_for_reconstruct>*> vars_on_face,
    const F0& reconstruct_lower_neighbor, const F1& reconstruct_upper_neighbor,
    const Variables<PrimsTags>& subcell_volume_prims,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>& eos,
    const Element<3>& element,
    const DirectionalIdMap<3, evolution::dg::subcell::GhostData>& ghost_data,
    const Mesh<3>& subcell_mesh, const Direction<3>& direction_to_reconstruct,
    const size_t ghost_zone_size, bool compute_conservatives);
}  // namespace grmhd::ValenciaDivClean::fd
