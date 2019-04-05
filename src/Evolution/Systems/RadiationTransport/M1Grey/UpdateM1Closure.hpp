// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/M1Closure.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
// IWYU pragma: no_forward_declare db::DataBox
/// \endcond

namespace Actions {

/// \ingroup ActionsGroup
/// \brief Compute the M1 closure and derived neutrino moments
///
/// \note `Metavariables` must specify a
/// `list_of_neutrino_species`.
///
/// Uses:
/// - DataBox: Items in RadiationTransport::M1Grey::M1Closure::argument_tags
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: RadiationTransport::M1Grey::M1Closure::return_tags
struct UpdateM1Closure {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<tmpl::size<DbTagsList>::value != 0> = nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using Closure = typename RadiationTransport::M1Grey::ComputeM1Closure<
        typename Metavariables::neutrino_species>;
    db::mutate_apply<Closure>(make_not_null(&box));
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions
