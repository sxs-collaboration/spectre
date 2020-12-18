// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
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
class GlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace evolution {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Compute and add the advective term for nonconservative systems on
 * moving meshes
 *
 * Adds the following term to the time derivative:
 *
 * \f{align}{
 *  v^i_g \partial_i u_\alpha,
 * \f}
 *
 * where \f$u_\alpha\f$ are the evolved variables and \f$v^i_g\f$ is the
 * velocity of the mesh.
 *
 * \note The term is always added in the `Frame::Inertial` frame, and the plus
 * sign arises because we add it to the time derivative.
 *
 * Uses:
 * - DataBox:
 *   - `Tags::deriv<system::variables_tags, Dim, Frame::Inertial>`
 *   - `domain::Tags::MeshVelocity<Dim>`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `Tags::dt<system::variable_tags>`
 */
struct AddMeshVelocityNonconservative {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    using variables_tags =
        typename Metavariables::system::variables_tag::tags_list;
    const auto& mesh_velocity =
        db::get<::domain::Tags::MeshVelocity<Metavariables::volume_dim>>(box);

    if (static_cast<bool>(mesh_velocity)) {
      tmpl::for_each<variables_tags>([&box, &mesh_velocity](
                                         auto variables_tag_v) noexcept {
        using variable_tag = typename decltype(variables_tag_v)::type;
        using dt_variable_tag = ::Tags::dt<variable_tag>;
        using deriv_tag =
            ::Tags::deriv<variable_tag, tmpl::size_t<Metavariables::volume_dim>,
                          Frame::Inertial>;

        db::mutate<dt_variable_tag>(
            make_not_null(&box),
            [](const auto dt_var_ptr, const auto& deriv_tensor,
               const std::optional<tnsr::I<
                   DataVector, Metavariables::volume_dim, Frame::Inertial>>&
                   grid_velocity) noexcept {
              for (size_t storage_index = 0;
                   storage_index < deriv_tensor.size(); ++storage_index) {
                // We grab the `deriv_tensor_index`, which would be e.g.
                // `(i, a, b)`, so `(0, 2, 3)`
                const auto deriv_tensor_index =
                    deriv_tensor.get_tensor_index(storage_index);
                // Then we drop the derivative index (the first entry) to get
                // `(a, b)` (or `(2, 3)`)
                const auto tensor_index =
                    all_but_specified_element_of(deriv_tensor_index, 0);
                // Set `deriv_index` to `i` (or `0` in the example)
                const size_t deriv_index = gsl::at(deriv_tensor_index, 0);
                dt_var_ptr->get(tensor_index) +=
                    grid_velocity->get(deriv_index) *
                    deriv_tensor[storage_index];
              }
            },
            db::get<deriv_tag>(box), mesh_velocity);
      });
    }
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace evolution
