// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Importers/Tags.hpp"
#include "Parallel/AlgorithmMetafunctions.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace importers::Actions {

/*!
 * \brief Wait for data from a volume data file to arrive and directly move it
 * into the DataBox
 *
 * Monitors `importers::Tags::VolumeData` in the element's inbox and moves the
 * received data directly into the `FieldTagsList` in the DataBox.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <typename ImporterOptionsGroup, typename FieldTagsList>
struct ReceiveVolumeData {
  using inbox_tags =
      tmpl::list<Tags::VolumeData<ImporterOptionsGroup, FieldTagsList>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, Parallel::AlgorithmExecution>
  apply(db::DataBox<DbTagsList>& box,
        tuples::TaggedTuple<InboxTags...>& inboxes,
        const Parallel::GlobalCache<Metavariables>& /*cache*/,
        const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
        const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox =
        tuples::get<Tags::VolumeData<ImporterOptionsGroup, FieldTagsList>>(
            inboxes);
    // Using `0` for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    const auto received_data = inbox.find(0_st);
    if (received_data == inbox.end()) {
      return {std::move(box), Parallel::AlgorithmExecution::Retry};
    }

    auto& element_data = received_data->second;
    tmpl::for_each<FieldTagsList>([&box, &element_data](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      db::mutate<tag>(
          make_not_null(&box),
          [&element_data](
              const gsl::not_null<typename tag::type*> value) noexcept {
            *value = std::move(tuples::get<tag>(element_data));
          });
    });
    inbox.erase(received_data);
    return {std::move(box), Parallel::AlgorithmExecution::Continue};
  }
};

}  // namespace importers::Actions
