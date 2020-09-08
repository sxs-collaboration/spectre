// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/ElementDataReaderActions.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace importers::Actions {

/// \cond
struct RegisterElementWithSelf;
/// \endcond

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
      const ParallelComponent* const /*meta*/) noexcept {
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
 * \brief Read a volume data file and distribute the data to all registered
 * elements.
 *
 * Invoke this action on the elements of an array parallel component to dispatch
 * reading the volume data file specified by the options in
 * `ImporterOptionsGroup`. The tensors in `FieldTagsList` will be loaded from
 * the file and distributed to all elements that have previously registered. Use
 * `importers::Actions::RegisterWithElementDataReader` to register the elements
 * of the array parallel component in a previous phase.
 *
 * Note that the volume data file will only be read once per node, triggered by
 * the first element that invokes this action. All subsequent invocations of
 * this action on the node will do nothing. See
 * `importers::Actions::ReadAllVolumeDataAndDistribute` for details.
 *
 * The data is distributed to the elements using `Parallel::receive_data`. The
 * elements can monitor `importers::Tags::VolumeData` in their inbox to wait for
 * the data and process it once it's available. We provide the action
 * `importers::Actions::ReceiveVolumeData` that waits for the data and moves it
 * directly into the DataBox. You can also implement a specialized action that
 * might verify and post-process the data before populating the DataBox.
 *
 * \see Dev guide on \ref dev_guide_importing
 */
template <typename ImporterOptionsGroup, typename FieldTagsList>
struct ReadVolumeData {
  using const_global_cache_tags =
      tmpl::list<Tags::FileName<ImporterOptionsGroup>,
                 Tags::Subgroup<ImporterOptionsGroup>,
                 Tags::ObservationValue<ImporterOptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& local_reader_component =
        *Parallel::get_parallel_component<
             importers::ElementDataReader<Metavariables>>(cache)
             .ckLocalBranch();
    Parallel::simple_action<importers::Actions::ReadAllVolumeDataAndDistribute<
        ImporterOptionsGroup, FieldTagsList, ParallelComponent>>(
        local_reader_component);
    return {std::move(box)};
  }
};

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
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    auto& inbox =
        tuples::get<Tags::VolumeData<ImporterOptionsGroup, FieldTagsList>>(
            inboxes);
    // Using `0` for the temporal ID since we only read the volume data once, so
    // there's no need to keep track of the temporal ID.
    const auto received_data = inbox.find(0_st);
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
    return {std::move(box)};
  }

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex>
  static bool is_ready(const db::DataBox<DbTags>& /*box*/,
                       const tuples::TaggedTuple<InboxTags...>& inboxes,
                       const Parallel::GlobalCache<Metavariables>& /*cache*/,
                       const ArrayIndex& /*array_index*/) noexcept {
    const auto& inbox =
        tuples::get<Tags::VolumeData<ImporterOptionsGroup, FieldTagsList>>(
            inboxes);
    return inbox.find(0_st) != inbox.end();
  }
};

}  // namespace importers::Actions
