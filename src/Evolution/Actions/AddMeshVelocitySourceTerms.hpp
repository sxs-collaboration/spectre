// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/optional.hpp>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
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

namespace evolution {
namespace Actions {
/*!
 * \ingroup ActionsGroup
 * \brief Compute and add the source term modification for moving meshes
 *
 * Adds to the time derivative *not* the source terms because some systems do
 * not have source terms and so we optimize for that. The term being added to
 * the time derivative is:
 *
 * \f{align}{
 *  -u_\alpha \partial_i v^i_g,
 * \f}
 *
 * where \f$u_\alpha\f$ are the evolved variables and \f$v^i_g\f$ is the
 * velocity of the mesh.
 *
 * Uses:
 * - DataBox:
 *   - `System::variables_tags`
 *   - `domain::Tags::DivFrameVelocity`
 *
 * DataBox changes:
 * - Adds: nothing
 * - Removes: nothing
 * - Modifies: `Tags::dt<system::variable_tags>`
 */
struct AddMeshVelocitySourceTerms {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT const
    using variables_tag = typename Metavariables::system::variables_tag;
    using dt_variables_tag = db::add_tag_prefix<Tags::dt, variables_tag>;
    db::mutate<dt_variables_tag>(
        make_not_null(&box),
        [](const auto dt_vars_ptr, const auto& vars,
           const boost::optional<Scalar<DataVector>>&
               div_grid_velocity) noexcept {
          if (div_grid_velocity) {
            *dt_vars_ptr -= vars * get(*div_grid_velocity);
          }
        },
        db::get<variables_tag>(box),
        db::get<domain::Tags::DivMeshVelocity>(box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
}  // namespace evolution
