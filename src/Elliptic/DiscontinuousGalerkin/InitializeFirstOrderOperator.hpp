// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct ConstGlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

namespace elliptic {
namespace dg {
namespace Actions {

/*!
 * \brief Initialize DataBox tags for building the first-order elliptic DG
 * operator
 */
template <size_t Dim, typename FluxesComputer, typename SourcesComputer,
          typename VariablesTag, typename PrimalVariables,
          typename AuxiliaryVariables>
struct InitializeFirstOrderOperator {
 private:
  static constexpr size_t volume_dim = Dim;
  using vars_tag = VariablesTag;
  using exterior_vars_tag = domain::Tags::Interface<
      domain::Tags::BoundaryDirectionsExterior<volume_dim>, vars_tag>;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;
  using inv_jacobian_tag =
      domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Inertial>;

  template <typename Directions>
  using face_tags =
      tmpl::list<domain::Tags::Slice<Directions, vars_tag>,
                 domain::Tags::Slice<Directions, fluxes_tag>,
                 domain::Tags::Slice<Directions, div_fluxes_tag>,
                 // For the strong first-order DG scheme we also need the
                 // interface normal dotted into the fluxes
                 domain::Tags::InterfaceCompute<
                     Directions, ::Tags::NormalDotFluxCompute<
                                     vars_tag, volume_dim, Frame::Inertial>>>;

  using fluxes_compute_tag =
      elliptic::Tags::FirstOrderFluxesCompute<volume_dim, FluxesComputer,
                                              vars_tag, PrimalVariables,
                                              AuxiliaryVariables>;
  using sources_compute_tag = elliptic::Tags::FirstOrderSourcesCompute<
      SourcesComputer, vars_tag, PrimalVariables, AuxiliaryVariables>;

  using exterior_tags = tmpl::list<
      // On exterior (ghost) boundary faces we compute the fluxes from the
      // data that is being set there to impose boundary conditions. Then, we
      // compute their normal-dot-fluxes. The flux divergences are sliced from
      // the volume.
      domain::Tags::InterfaceCompute<
          domain::Tags::BoundaryDirectionsExterior<volume_dim>,
          fluxes_compute_tag>,
      domain::Tags::InterfaceCompute<
          domain::Tags::BoundaryDirectionsExterior<volume_dim>,
          ::Tags::NormalDotFluxCompute<vars_tag, volume_dim, Frame::Inertial>>,
      domain::Tags::Slice<domain::Tags::BoundaryDirectionsExterior<volume_dim>,
                          div_fluxes_tag>>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // Initialize the variables on exterior (ghost) boundary faces. They are
    // updated throughout the algorithm to impose boundary conditions.
    db::item_type<exterior_vars_tag> exterior_boundary_vars{};
    const auto& mesh = db::get<domain::Tags::Mesh<volume_dim>>(box);
    for (const auto& direction : db::get<domain::Tags::Element<volume_dim>>(box)
                                     .external_boundaries()) {
      exterior_boundary_vars[direction] = db::item_type<vars_tag>{
          mesh.slice_away(direction.dimension()).number_of_grid_points()};
    }
    using compute_tags = tmpl::flatten<tmpl::list<
        fluxes_compute_tag, sources_compute_tag,
        ::Tags::DivVariablesCompute<fluxes_tag, inv_jacobian_tag>,
        face_tags<domain::Tags::InternalDirections<volume_dim>>,
        face_tags<domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
        exterior_tags>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeFirstOrderOperator, db::AddSimpleTags<exterior_vars_tag>,
            compute_tags>(std::move(box), std::move(exterior_boundary_vars)));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
