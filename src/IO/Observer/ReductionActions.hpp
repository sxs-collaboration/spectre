// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/Assert.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {

/// \cond
template <class Metavariables>
struct ObserverWriter;
/// \endcond

namespace ThreadedActions {
/// \cond
struct WriteReductionData;
/// \endcond
}  // namespace ThreadedActions

namespace Actions {
/// \cond
struct ContributeReductionDataToWriter;
/// \endcond

/*!
 * \ingroup ObserverGroup
 * \brief Send reduction data to the observer group.
 *
 * Once everything at a specific `ObservationId` has been contributed to the
 * reduction, the groups reduce to their local nodegroup.
 *
 * The caller of this Action (which is to be invoked on the Observer parallel
 * component) must pass in an `observation_id` used to uniquely identify the
 * observation in time, the name of the `h5::Dat` subfile in the HDF5 file (e.g.
 * `/element_data`, where the slash is important), a `std::vector<std::string>`
 * of names of the quantities being reduced (e.g. `{"Time", "L1ErrorDensity",
 * "L2ErrorDensity"}`), and the `Parallel::ReductionData` that holds the
 * `ReductionDatums` containing info on how to do the reduction.
 *
 * The observer components need to know all expected reduction data types by
 * compile-time, so they rely on the
 * `Metavariables::observed_reduction_data_tags` alias to collect them in one
 * place. To this end, each Action that contributes reduction data must expose
 * the type alias as:
 *
 * \snippet ObserverHelpers.hpp make_reduction_data_tags
 *
 * Then, in the `Metavariables` collect them from all observing Actions like
 * this:
 *
 * \snippet tests/Unit/Elliptic/Systems/Poisson/Actions/Test_Observe.cpp collect_reduction_data_tags
 */
struct ContributeReductionData {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... Ts,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    db::mutate<Tags::ReductionData<Ts...>, Tags::ReductionDataNames<Ts...>,
               Tags::ReductionObserversContributed>(
        make_not_null(&box),
        [
          &observation_id, reduction_data = std::move(reduction_data),
          &reduction_names, &cache, &subfile_name
        ](const gsl::not_null<std::unordered_map<
              ObservationId, Parallel::ReductionData<Ts...>>*>
              reduction_data_map,
          const gsl::not_null<
              std::unordered_map<ObservationId, std::vector<std::string>>*>
              reduction_names_map,
          const gsl::not_null<
              std::unordered_map<observers::ObservationId, size_t>*>
              reduction_observers_contributed,
          const std::unordered_set<ArrayComponentId>&
              reduction_component_ids) mutable noexcept {
          auto& contribute_count =
              (*reduction_observers_contributed)[observation_id];
          if (reduction_data_map->count(observation_id) == 0) {
            reduction_data_map->emplace(observation_id,
                                        std::move(reduction_data));
            reduction_names_map->emplace(observation_id, reduction_names);
            contribute_count = 1;
          } else {
            ASSERT(
                reduction_names_map->at(observation_id) == reduction_names,
                "Reduction names differ at ObservationId "
                    << observation_id
                    << " with the expected names being "
                    // Use MakeString to get around ADL for STL stream operators
                    // (MakeString is in global namespace).
                    << (MakeString{} << reduction_names_map->at(observation_id)
                                     << " and the received names being "
                                     << reduction_names));
            reduction_data_map->operator[](observation_id)
                .combine(std::move(reduction_data));
            contribute_count++;
          }

          // Check if we have received all reduction data from the registered
          // elements. If so, we reduce to the local ObserverWriter nodegroup.
          if (contribute_count == reduction_component_ids.size()) {
            const auto node_id = Parallel::my_node();
            auto& local_writer = *Parallel::get_parallel_component<
                                      ObserverWriter<Metavariables>>(cache)
                                      .ckLocalBranch();
            Parallel::threaded_action<ThreadedActions::WriteReductionData>(
                local_writer, observation_id, subfile_name,
                node_id == 0 ? std::move((*reduction_names_map)[observation_id])
                             : std::vector<std::string>{},
                std::move((*reduction_data_map)[observation_id]));
            reduction_data_map->erase(observation_id);
            reduction_names_map->erase(observation_id);
            reduction_observers_contributed->erase(observation_id);
          }
        },
        db::get<Tags::ReductionArrayComponentIds>(box));
  }
};
}  // namespace Actions

namespace ThreadedActions {
/*!
 * \ingroup ObserverGroup
 * \brief Collect the reduction data from the Observer group on the
 * ObserverWriter nodegroup before sending to node 0 for writing to disk.
 *
 * \note This action is also used for writing on node 0.
 */
struct WriteReductionData {
 private:
  template <typename... Ts, size_t... Is>
  static void write_data(const std::string& subfile_name,
                         std::vector<std::string>&& legend,
                         std::tuple<Ts...>&& data,
                         const std::string& file_prefix,
                         std::index_sequence<Is...> /*meta*/) noexcept {
    static_assert(sizeof...(Ts) > 0,
                  "Must be reducing at least one piece of data");
    std::vector<double> data_to_append{
        static_cast<double>(std::get<Is>(data))...};

    h5::H5File<h5::AccessType::ReadWrite> h5file(file_prefix + ".h5", true);
    constexpr size_t version_number = 0;
    auto& time_series_file = h5file.try_insert<h5::Dat>(
        subfile_name, std::move(legend), version_number);
    time_series_file.append(data_to_append);
  }

 public:
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, typename... ReductionDatums,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const gsl::not_null<CmiNodeLock*> node_lock,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    std::vector<std::string>&& reduction_names,
                    Parallel::ReductionData<ReductionDatums...>&&
                        in_reduction_data) noexcept {
    CmiNodeLock file_lock;
    bool write_to_disk = false;
    std::vector<std::string> legend{};
    Parallel::lock(node_lock);
    const auto expected_calls_from_this_node =
        [&box, &observation_id ]() noexcept {
      const auto hash = observation_id.observation_type_hash();
      const auto& registered_reduction_observers =
          db::get<Tags::ReductionObserversRegistered>(box);
      return (registered_reduction_observers.count(hash) == 1)
                 ? registered_reduction_observers.at(hash).size()
                 : 0;
    }
    ();
    const auto expected_calls_from_other_nodes =
        [&box, &observation_id ]() noexcept {
      const auto hash = observation_id.observation_type_hash();
      const auto& registered_reduction_observers =
          db::get<Tags::ReductionObserversRegisteredNodes>(box);
      return (registered_reduction_observers.count(hash) == 1)
                 ? registered_reduction_observers.at(hash).size()
                 : 0;
    }
    ();
    db::mutate<Tags::ReductionData<ReductionDatums...>,
               Tags::ReductionDataNames<ReductionDatums...>,
               Tags::ReductionObserversContributed, Tags::H5FileLock>(
        make_not_null(&box),
        [
          &cache, &expected_calls_from_this_node,
          &expected_calls_from_other_nodes, &file_lock, &in_reduction_data,
          &legend, &observation_id, &reduction_names, &subfile_name,
          &write_to_disk
        ](const gsl::not_null<
              db::item_type<Tags::ReductionData<ReductionDatums...>>*>
              reduction_data,
          const gsl::not_null<
              std::unordered_map<ObservationId, std::vector<std::string>>*>
              reduction_names_map,
          const gsl::not_null<
              std::unordered_map<observers::ObservationId, size_t>*>
              reduction_observers_contributed,
          const gsl::not_null<CmiNodeLock*> reduction_file_lock) noexcept {
          auto& contribute_count =
              (*reduction_observers_contributed)[observation_id];
          const auto node_id = Parallel::my_node();

          if (node_id == 0 and not reduction_names.empty()) {
            reduction_names_map->emplace(observation_id,
                                         std::move(reduction_names));
          }

          if (UNLIKELY(node_id == 0 and expected_calls_from_other_nodes == 0 and
                       expected_calls_from_this_node == 1)) {
            // Here this Action will be called only once, so we take
            // a shortcut and just write to disk.
            write_to_disk = true;
            file_lock = *reduction_file_lock;
            legend = std::move(reduction_names_map->operator[](observation_id));
            reduction_names_map->erase(observation_id);
          } else if (reduction_data->count(observation_id) == 0) {
            // This Action has been called for the first time,
            // so all we need to do is move the input data to the
            // reduction_data in the DataBox.
            reduction_data->operator[](observation_id) =
                std::move(in_reduction_data);
            contribute_count = 1;
          } else if (node_id == 0 and
                     contribute_count == expected_calls_from_this_node +
                                             expected_calls_from_other_nodes -
                                             1) {
            // This is the final time this Action is called on node 0.
            // The `-1` in the above `if` statement is because on the final
            // step, contribute_count (which was incremented at the end of each
            // previous call) should be one less than the expected number of
            // calls.
            in_reduction_data.combine(
                std::move(reduction_data->operator[](observation_id)));
            reduction_data->erase(observation_id);
            reduction_observers_contributed->erase(observation_id);
            write_to_disk = true;
            file_lock = *reduction_file_lock;
            legend = std::move(reduction_names_map->operator[](observation_id));
            reduction_names_map->erase(observation_id);
          } else {
            // This Action is being called at least the second time
            // (but not the final time if on node 0).
            reduction_data->at(observation_id)
                .combine(std::move(in_reduction_data));
            contribute_count++;
          }

          // Check if we have received all reduction data from the Observer
          // group. If so we reduce to node 0 for writing to disk.
          if (node_id != 0 and
              reduction_observers_contributed->at(observation_id) ==
                  expected_calls_from_this_node) {
            Parallel::threaded_action<WriteReductionData>(
                Parallel::get_parallel_component<ObserverWriter<Metavariables>>(
                    cache)[0],
                observation_id, subfile_name, std::vector<std::string>{},
                std::move(reduction_data->operator[](observation_id)));
            reduction_observers_contributed->erase(observation_id);
            reduction_data->erase(observation_id);
          }
        });
    Parallel::unlock(node_lock);

    if (write_to_disk) {
      Parallel::lock(&file_lock);
      in_reduction_data.finalize();
      WriteReductionData::write_data(
          subfile_name, std::move(legend), std::move(in_reduction_data.data()),
          Parallel::get<OptionTags::ReductionFileName>(cache),
          std::make_index_sequence<sizeof...(ReductionDatums)>{});
      Parallel::unlock(&file_lock);
    }
  }
};
}  // namespace ThreadedActions
}  // namespace observers
