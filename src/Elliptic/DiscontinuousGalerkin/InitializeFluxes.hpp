// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/FirstOrderComputeTags.hpp"
#include "Elliptic/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
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
 * \brief Initialize DataBox tags related to discontinuous Galerkin fluxes
 *
 * Uses:
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *   - `fluxes`
 */
template <typename Metavariables>
struct InitializeFluxes {
 private:
  using system = typename Metavariables::system;
  static constexpr size_t volume_dim = system::volume_dim;
  using vars_tag = typename system::variables_tag;
  using fluxes_tag =
      db::add_tag_prefix<::Tags::Flux, vars_tag, tmpl::size_t<volume_dim>,
                         Frame::Inertial>;
  using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag>;

  template <typename Directions>
  using face_tags =
      tmpl::list<domain::Tags::Slice<Directions, fluxes_tag>,
                 domain::Tags::Slice<Directions, div_fluxes_tag>,
                 // For the strong first-order DG scheme we also need the
                 // interface normal dotted into the fluxes
                 domain::Tags::InterfaceCompute<
                     Directions, ::Tags::NormalDotFluxCompute<
                                     vars_tag, volume_dim, Frame::Inertial>>>;

  using fluxes_compute_tag = elliptic::Tags::FirstOrderFluxesCompute<
      volume_dim, typename system::fluxes, typename system::variables_tag,
      typename system::primal_variables, typename system::auxiliary_variables>;

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
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = tmpl::flatten<tmpl::list<
        face_tags<domain::Tags::InternalDirections<volume_dim>>,
        face_tags<domain::Tags::BoundaryDirectionsInterior<volume_dim>>,
        exterior_tags>>;
    return std::make_tuple(
        ::Initialization::merge_into_databox<InitializeFluxes,
                                             db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};
}  // namespace Actions
}  // namespace dg
}  // namespace elliptic
