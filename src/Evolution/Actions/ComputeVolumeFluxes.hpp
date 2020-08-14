// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Compute the volume fluxes of the evolved variables
 *
 * The mesh velocity is added to the flux automatically if the mesh is moving.
 * That is,
 *
 * \f{align}{
 *  F^i_{\alpha}\to F^i_{\alpha}-v^i_{g} u_{\alpha}
 * \f}
 *
 * where \f$F^i_{\alpha}\f$ are the fluxes when the mesh isn't moving,
 * \f$v^i_g\f$ is the velocity of the mesh, and \f$u_{\alpha}\f$ are the evolved
 * variables.
 *
 * Uses:
 * - DataBox:
 *   - Items in system::volume_fluxes::argument_tags
 *   - `domain::Tags::MeshVelocity<Metavariables::volume_dim>`
 *   - `Metavariables::system::variables_tag`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies:
 *   db::add_tag_prefix<Tags::Flux, variables_tag,
 *                      tmpl::size_t<system::volume_dim>, Frame::Inertial>
 */
struct ComputeVolumeFluxes {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    db::mutate_apply<typename Metavariables::system::volume_fluxes>(
        make_not_null(&box));
    const auto use_moving_mesh = static_cast<bool>(
        db::get<domain::Tags::MeshVelocity<Metavariables::volume_dim>>(box));
    if (use_moving_mesh) {
      using variables_tag = typename Metavariables::system::variables_tag;
      using fluxes_tag =
          db::add_tag_prefix<::Tags::Flux, variables_tag,
                             tmpl::size_t<Metavariables::volume_dim>,
                             Frame::Inertial>;
      db::mutate<fluxes_tag>(
          make_not_null(&box),
          [](const auto fluxes_ptr, const auto& evolved_vars,
             const tnsr::I<DataVector, Metavariables::volume_dim,
                           Frame::Inertial>& grid_velocity) noexcept {
            tmpl::for_each<typename fluxes_tag::tags_list>(
                [&evolved_vars, &fluxes_ptr, &
                 grid_velocity ](auto tag_v) noexcept {
                  using ::get;
                  using flux_tag = typename decltype(tag_v)::type;
                  using var_tag = db::remove_tag_prefix<flux_tag>;
                  auto& flux_tensor = get<flux_tag>(*fluxes_ptr);
                  for (size_t storage_index = 0;
                       storage_index < flux_tensor.size(); ++storage_index) {
                    const auto flux_tensor_index =
                        flux_tensor.get_tensor_index(storage_index);
                    const auto tensor_index =
                        all_but_specified_element_of(flux_tensor_index, 0);
                    const size_t flux_index = gsl::at(flux_tensor_index, 0);

                    flux_tensor[storage_index] -=
                        get<var_tag>(evolved_vars).get(tensor_index) *
                        grid_velocity.get(flux_index);
                  }
                });
          },
          db::get<variables_tag>(box),
          *db::get<domain::Tags::MeshVelocity<Metavariables::volume_dim>>(box));
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
