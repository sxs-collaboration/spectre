// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace importers {
template <typename Metavariables>
struct ElementDataReader;
namespace Actions {
struct RegisterElementWithSelf;
}  // namespace Actions
}  // namespace importers
/// \endcond

namespace importers::Actions {

/*!
 * \brief Register an element with the volume data reader component.
 *
 * Invoke this action on each element of an array parallel component to register
 * them for receiving imported volume data.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
struct RegisterWithElementDataReader {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const std::string element_name = MakeString{}
                                     << ElementId<Dim>(array_index);
    auto& local_reader_component =
        *Parallel::get_parallel_component<
             importers::ElementDataReader<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<importers::Actions::RegisterElementWithSelf>(
        local_reader_component,
        observers::ArrayComponentId(
            std::add_pointer_t<ParallelComponent>{nullptr},
            Parallel::ArrayIndex<ElementId<Dim>>(array_index)),
        element_name);
    return {std::move(box)};
  }
};

/*!
 * \brief Invoked on the `importers::ElementDataReader` component to store the
 * registered data.
 *
 * The `importers::Actions::RegisterWithElementDataReader` action, which is
 * performed on each element of an array parallel component, invokes this action
 * on the `importers::ElementDataReader` component.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
struct RegisterElementWithSelf {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, typename DataBox = db::DataBox<DbTagsList>,
      Requires<db::tag_is_retrievable_v<Tags::RegisteredElements, DataBox>> =
          nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const observers::ArrayComponentId& array_component_id,
                    const std::string& grid_name) {
    db::mutate<Tags::RegisteredElements>(
        make_not_null(&box),
        [&array_component_id, &grid_name](
            const gsl::not_null<
                std::unordered_map<observers::ArrayComponentId, std::string>*>
                registered_elements) {
          (*registered_elements)[array_component_id] = grid_name;
        });
  }
};

}  // namespace importers::Actions
