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
#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/PrettyType.hpp"
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
struct CollectReductionDataOnNode;
struct WriteReductionData;
/// \endcond
}  // namespace ThreadedActions

namespace Actions {
/// \cond
struct ContributeReductionDataToWriter;
/// \endcond

/*!
 * \ingroup ObserversGroup
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
 * Then, in the `Metavariables` collect them from all observing Actions using
 * the `observers::collect_reduction_data_tags` metafunction.
 */
struct ContributeReductionData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename... Ts>
  static auto apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const observers::ObservationId& observation_id,
                    const ArrayComponentId& sender_array_id,
                    const std::string& subfile_name,
                    const std::vector<std::string>& reduction_names,
                    Parallel::ReductionData<Ts...>&& reduction_data) noexcept {
    if constexpr (tmpl::list_contains_v<DbTagsList,
                                        Tags::ReductionData<Ts...>> and
                  tmpl::list_contains_v<DbTagsList,
                                        Tags::ReductionDataNames<Ts...>> and
                  tmpl::list_contains_v<DbTagsList,
                                        Tags::ReductionsContributed>) {
      db::mutate<Tags::ReductionData<Ts...>, Tags::ReductionDataNames<Ts...>,
                 Tags::ReductionsContributed>(
          make_not_null(&box),
          [&array_index, &cache, &observation_id,
           reduction_data = std::move(reduction_data), &reduction_names,
           &sender_array_id, &subfile_name](
              const gsl::not_null<std::unordered_map<
                  ObservationId, Parallel::ReductionData<Ts...>>*>
                  reduction_data_map,
              const gsl::not_null<
                  std::unordered_map<ObservationId, std::vector<std::string>>*>
                  reduction_names_map,
              const gsl::not_null<std::unordered_map<
                  ObservationId, std::unordered_set<ArrayComponentId>>*>
                  reduction_observers_contributed,
              const std::unordered_map<ObservationKey,
                                       std::unordered_set<ArrayComponentId>>&
                  observations_registered) mutable noexcept {
            ASSERT(observations_registered.find(
                       observation_id.observation_key()) !=
                       observations_registered.end(),
                   "Couldn't find registration key "
                       << observation_id.observation_key()
                       << " in the registered observers.");

            auto& contributed_array_ids =
                (*reduction_observers_contributed)[observation_id];
            if (UNLIKELY(contributed_array_ids.find(sender_array_id) !=
                         contributed_array_ids.end())) {
              ERROR("Already received reduction data to observation id "
                    << observation_id << " from array component id "
                    << sender_array_id);
            }
            contributed_array_ids.insert(sender_array_id);

            if (reduction_data_map->count(observation_id) == 0) {
              reduction_data_map->emplace(observation_id,
                                          std::move(reduction_data));
              reduction_names_map->emplace(observation_id, reduction_names);
            } else {
              if (UNLIKELY(reduction_names_map->at(observation_id) !=
                           reduction_names)) {
                using ::operator<<;
                ERROR("Reduction names differ at ObservationId "
                      << observation_id << " with the expected names being "
                      << reduction_names_map->at(observation_id)
                      << " and the received names being " << reduction_names);
              }
              reduction_data_map->operator[](observation_id)
                  .combine(std::move(reduction_data));
            }

            // Check if we have received all reduction data from the registered
            // elements. If so, we reduce to the local ObserverWriter nodegroup.
            if (UNLIKELY(
                    contributed_array_ids.size() ==
                    observations_registered.at(observation_id.observation_key())
                        .size())) {
              auto& local_writer = *Parallel::get_parallel_component<
                                        ObserverWriter<Metavariables>>(cache)
                                        .ckLocalBranch();
              Parallel::threaded_action<
                  ThreadedActions::CollectReductionDataOnNode>(
                  local_writer, observation_id,
                  ArrayComponentId{
                      std::add_pointer_t<ParallelComponent>{nullptr},
                      Parallel::ArrayIndex<ArrayIndex>(array_index)},
                  subfile_name, (*reduction_names_map)[observation_id],
                  std::move((*reduction_data_map)[observation_id]));
              reduction_data_map->erase(observation_id);
              reduction_names_map->erase(observation_id);
              reduction_observers_contributed->erase(observation_id);
            }
          },
          db::get<Tags::ExpectedContributorsForObservations>(box));
    } else {
      ERROR("Could not find the tag "
            << pretty_type::get_name<Tags::ReductionData<Ts...>>() << ' '
            << pretty_type::get_name<Tags::ReductionDataNames<Ts...>>()
            << " or " << pretty_type::get_name<Tags::ReductionsContributed>()
            << " in the DataBox.");
    }
  }
};
}  // namespace Actions

namespace ThreadedActions {
/*!
 * \brief Gathers all the reduction data from all processing elements/cores on a
 * node.
 */
struct CollectReductionDataOnNode {
 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename... ReductionDatums>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const observers::ObservationId& observation_id,
                    ArrayComponentId observer_group_id,
                    const std::string& subfile_name,
                    std::vector<std::string>&& reduction_names,
                    Parallel::ReductionData<ReductionDatums...>&&
                        received_reduction_data) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ReductionData<ReductionDatums...>> and
                  tmpl::list_contains_v<DbTagsList, Tags::ReductionDataNames<
                                                        ReductionDatums...>> and
                  tmpl::list_contains_v<DbTagsList, Tags::ReductionDataLock>) {
      // The below gymnastics with pointers is done in order to minimize the
      // time spent locking the entire node, which is necessary because the
      // DataBox does not allow any functions calls, both get and mutate, during
      // a mutate. This design choice in DataBox is necessary to guarantee a
      // consistent state throughout mutation. Here, however, we need to be
      // reasonable efficient in parallel and so we manually guarantee that
      // consistent state. To this end, we create pointers and assign to them
      // the data in the DataBox which is guaranteed to be pointer stable. The
      // data itself is guaranteed to be stable inside the ReductionDataLock.
      std::unordered_map<observers::ObservationId,
                         Parallel::ReductionData<ReductionDatums...>>*
          reduction_data = nullptr;
      std::unordered_map<ObservationId, std::vector<std::string>>*
          reduction_names_map = nullptr;
      std::unordered_map<observers::ObservationId,
                         std::unordered_set<ArrayComponentId>>*
          reduction_observers_contributed = nullptr;
      Parallel::NodeLock* reduction_data_lock = nullptr;
      size_t observations_registered_with_id =
          std::numeric_limits<size_t>::max();

      node_lock->lock();
      db::mutate<Tags::ReductionData<ReductionDatums...>,
                 Tags::ReductionDataNames<ReductionDatums...>,
                 Tags::ReductionsContributed, Tags::ReductionDataLock>(
          make_not_null(&box),
          [&reduction_data, &reduction_names_map,
           &reduction_observers_contributed, &reduction_data_lock,
           &observation_id, &observer_group_id,
           &observations_registered_with_id](
              const gsl::not_null<std::unordered_map<
                  observers::ObservationId,
                  Parallel::ReductionData<ReductionDatums...>>*>
                  reduction_data_ptr,
              const gsl::not_null<
                  std::unordered_map<ObservationId, std::vector<std::string>>*>
                  reduction_names_map_ptr,
              const gsl::not_null<
                  std::unordered_map<observers::ObservationId,
                                     std::unordered_set<ArrayComponentId>>*>
                  reduction_observers_contributed_ptr,
              const gsl::not_null<Parallel::NodeLock*> reduction_data_lock_ptr,
              const std::unordered_map<ObservationKey,
                                       std::unordered_set<ArrayComponentId>>&
                  observations_registered) noexcept {
            const ObservationKey& key{observation_id.observation_key()};
            const auto& registered_group_ids = observations_registered.at(key);
            if (UNLIKELY(registered_group_ids.find(observer_group_id) ==
                         registered_group_ids.end())) {
              ERROR("The observer group id "
                    << observer_group_id
                    << " was not registered for the observation id "
                    << observation_id);
            }
            reduction_data = &*reduction_data_ptr;
            reduction_names_map = &*reduction_names_map_ptr;
            reduction_observers_contributed =
                &*reduction_observers_contributed_ptr;
            reduction_data_lock = &*reduction_data_lock_ptr;
            observations_registered_with_id =
                observations_registered.at(key).size();
          },
          db::get<Tags::ExpectedContributorsForObservations>(box));
      node_lock->unlock();

      ASSERT(
          observations_registered_with_id != std::numeric_limits<size_t>::max(),
          "Failed to set observations_registered_with_id when mutating the "
          "DataBox. This is a bug in the code.");

      // Now that we've retrieved pointers to the data in the DataBox we wish to
      // manipulate, lock the data and manipulate it.
      reduction_data_lock->lock();
      auto& contributed_group_ids =
          (*reduction_observers_contributed)[observation_id];

      if (UNLIKELY(contributed_group_ids.find(observer_group_id) !=
                   contributed_group_ids.end())) {
        ERROR("Already received reduction data to observation id "
              << observation_id << " from array component id "
              << observer_group_id);
      }
      contributed_group_ids.insert(observer_group_id);

      if (reduction_data->find(observation_id) == reduction_data->end()) {
        // This Action has been called for the first time,
        // so all we need to do is move the input data to the
        // reduction_data in the DataBox.
        reduction_data->operator[](observation_id) =
            std::move(received_reduction_data);
      } else {
        // This Action is being called at least the second time
        // (but not the final time if on node 0).
        reduction_data->at(observation_id)
            .combine(std::move(received_reduction_data));
      }

      if (UNLIKELY(reduction_names.empty())) {
        ERROR(
            "The reduction names, which is a std::vector of the names of "
            "the columns in the file, must be non-empty.");
      }
      if (auto current_names = reduction_names_map->find(observation_id);
          current_names == reduction_names_map->end()) {
        reduction_names_map->emplace(observation_id,
                                     std::move(reduction_names));
      } else if (UNLIKELY(current_names->second != reduction_names)) {
        ERROR(
            "The reduction names passed in must match the currently "
            "known reduction names.");
      }

      // Check if we have received all reduction data from the Observer
      // group. If so we reduce to node 0 for writing to disk. We use a bool
      // `send_data` to allow us to defer the send call until after we've
      // unlocked the lock.
      bool send_data = false;
      if (reduction_observers_contributed->at(observation_id).size() ==
          observations_registered_with_id) {
        send_data = true;
        // We intentionally move the data out of the map and erase it
        // before call `WriteReductionData` since if the call to
        // `WriteReductionData` is inlined and we erase data from the maps
        // afterward we would lose data.
        reduction_names =
            std::move(reduction_names_map->operator[](observation_id));
        received_reduction_data =
            std::move(reduction_data->operator[](observation_id));
        reduction_observers_contributed->erase(observation_id);
        reduction_data->erase(observation_id);
        reduction_names_map->erase(observation_id);
      }
      reduction_data_lock->unlock();

      if (send_data) {
        Parallel::threaded_action<WriteReductionData>(
            Parallel::get_parallel_component<ObserverWriter<Metavariables>>(
                cache)[0],
            observation_id, static_cast<size_t>(Parallel::my_node()),
            subfile_name,
            // NOLINTNEXTLINE(bugprone-use-after-move)
            std::move(reduction_names), std::move(received_reduction_data));
      }
    } else {
      (void)node_lock;
      (void)observer_group_id;
      ERROR("Could not find one of the tags: "
            << pretty_type::get_name<Tags::ReductionData<ReductionDatums...>>()
            << ' '
            << pretty_type::get_name<
                   Tags::ReductionDataNames<ReductionDatums...>>()
            << " or Tags::ReductionDataLock");
    }
  }
};

/*!
 * \ingroup ObserversGroup
 * \brief Write reduction data to disk from node 0.
 */
struct WriteReductionData {
 private:
  static void append_to_reduction_data(
      const gsl::not_null<std::vector<double>*> all_reduction_data,
      const double t) noexcept {
    all_reduction_data->push_back(t);
  }

  static void append_to_reduction_data(
      const gsl::not_null<std::vector<double>*> all_reduction_data,
      const std::vector<double>& t) noexcept {
    all_reduction_data->insert(all_reduction_data->end(), t.begin(), t.end());
  }

  template <typename... Ts, size_t... Is>
  static void write_data(const std::string& subfile_name,
                         std::vector<std::string>&& legend,
                         std::tuple<Ts...>&& data,
                         const std::string& file_prefix,
                         std::index_sequence<Is...> /*meta*/) noexcept {
    static_assert(sizeof...(Ts) > 0,
                  "Must be reducing at least one piece of data");
    std::vector<double> data_to_append{};
    EXPAND_PACK_LEFT_TO_RIGHT(
        append_to_reduction_data(&data_to_append, std::get<Is>(data)));

    h5::H5File<h5::AccessType::ReadWrite> h5file(file_prefix + ".h5", true);
    constexpr size_t version_number = 0;
    auto& time_series_file = h5file.try_insert<h5::Dat>(
        subfile_name, std::move(legend), version_number);
    time_series_file.append(data_to_append);
  }

 public:
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename... ReductionDatums>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const observers::ObservationId& observation_id,
                    const size_t sender_node_number,
                    const std::string& subfile_name,
                    std::vector<std::string>&& reduction_names,
                    Parallel::ReductionData<ReductionDatums...>&&
                        received_reduction_data) noexcept {
    if constexpr (tmpl::list_contains_v<
                      DbTagsList, Tags::ReductionData<ReductionDatums...>> and
                  tmpl::list_contains_v<DbTagsList, Tags::ReductionDataNames<
                                                        ReductionDatums...>> and
                  tmpl::list_contains_v<
                      DbTagsList, Tags::NodesThatContributedReductions> and
                  tmpl::list_contains_v<DbTagsList, Tags::ReductionDataLock> and
                  tmpl::list_contains_v<DbTagsList, Tags::H5FileLock>) {
      // The below gymnastics with pointers is done in order to minimize the
      // time spent locking the entire node, which is necessary because the
      // DataBox does not allow any functions calls, both get and mutate, during
      // a mutate. This design choice in DataBox is necessary to guarantee a
      // consistent state throughout mutation. Here, however, we need to be
      // reasonable efficient in parallel and so we manually guarantee that
      // consistent state. To this end, we create pointers and assign to them
      // the data in the DataBox which is guaranteed to be pointer stable. The
      // data itself is guaranteed to be stable inside the ReductionDataLock.
      std::unordered_map<observers::ObservationId,
                         Parallel::ReductionData<ReductionDatums...>>*
          reduction_data = nullptr;
      std::unordered_map<ObservationId, std::vector<std::string>>*
          reduction_names_map = nullptr;
      std::unordered_map<observers::ObservationId, std::unordered_set<size_t>>*
          nodes_contributed = nullptr;
      Parallel::NodeLock* reduction_data_lock = nullptr;
      Parallel::NodeLock* reduction_file_lock = nullptr;
      size_t observations_registered_with_id =
          std::numeric_limits<size_t>::max();

      node_lock->lock();
      db::mutate<Tags::ReductionData<ReductionDatums...>,
                 Tags::ReductionDataNames<ReductionDatums...>,
                 Tags::NodesThatContributedReductions, Tags::ReductionDataLock,
                 Tags::H5FileLock>(
          make_not_null(&box),
          [&nodes_contributed, &reduction_data, &reduction_names_map,
           &reduction_data_lock, &reduction_file_lock, &observation_id,
           &observations_registered_with_id, &sender_node_number](
              const gsl::not_null<
                  db::item_type<Tags::ReductionData<ReductionDatums...>>*>
                  reduction_data_ptr,
              const gsl::not_null<
                  std::unordered_map<ObservationId, std::vector<std::string>>*>
                  reduction_names_map_ptr,
              const gsl::not_null<std::unordered_map<
                  ObservationId, std::unordered_set<size_t>>*>
                  nodes_contributed_ptr,
              const gsl::not_null<Parallel::NodeLock*> reduction_data_lock_ptr,
              const gsl::not_null<Parallel::NodeLock*> reduction_file_lock_ptr,
              const std::unordered_map<ObservationKey, std::set<size_t>>&
                  nodes_registered_for_reductions) noexcept {
            const ObservationKey& key{observation_id.observation_key()};
            ASSERT(nodes_registered_for_reductions.find(key) !=
                       nodes_registered_for_reductions.end(),
                   "Performing reduction with unregistered ID key "
                       << observation_id.observation_key());
            const auto& registered_nodes =
                nodes_registered_for_reductions.at(key);

            if (UNLIKELY(registered_nodes.find(sender_node_number) ==
                         registered_nodes.end())) {
              ERROR("Node " << sender_node_number
                            << " was not registered for the observation id "
                            << observation_id);
            }

            reduction_data = &*reduction_data_ptr;
            reduction_names_map = &*reduction_names_map_ptr;
            nodes_contributed = &*nodes_contributed_ptr;
            reduction_data_lock = &*reduction_data_lock_ptr;
            reduction_file_lock = &*reduction_file_lock_ptr;
            observations_registered_with_id =
                nodes_registered_for_reductions.at(key).size();
          },
          db::get<Tags::NodesExpectedToContributeReductions>(box));
      node_lock->unlock();

      ASSERT(
          observations_registered_with_id != std::numeric_limits<size_t>::max(),
          "Failed to set observations_registered_with_id when mutating the "
          "DataBox. This is a bug in the code.");

      // Now that we've retrieved pointers to the data in the DataBox we wish to
      // manipulate, lock the data and manipulate it.
      reduction_data_lock->lock();
      auto& nodes_contributed_to_observation =
          (*nodes_contributed)[observation_id];
      if (nodes_contributed_to_observation.find(sender_node_number) !=
          nodes_contributed_to_observation.end()) {
        ERROR("Already received reduction data at observation id "
              << observation_id << " from node " << sender_node_number);
      }
      nodes_contributed_to_observation.insert(sender_node_number);

      if (UNLIKELY(reduction_names.empty())) {
        ERROR(
            "The reduction names, which is a std::vector of the names of "
            "the columns in the file, must be non-empty.");
      }
      if (auto current_names = reduction_names_map->find(observation_id);
          current_names == reduction_names_map->end()) {
        reduction_names_map->emplace(observation_id,
                                     std::move(reduction_names));
      } else if (UNLIKELY(current_names->second != reduction_names)) {
        using ::operator<<;
        ERROR(
            "The reduction names passed in must match the currently "
            "known reduction names. Current ones are "
            << current_names->second << " while the received are "
            << reduction_names);
      }

      if (reduction_data->find(observation_id) == reduction_data->end()) {
        // This Action has been called for the first time,
        // so all we need to do is move the input data to the
        // reduction_data in the DataBox.
        reduction_data->operator[](observation_id) =
            std::move(received_reduction_data);
      } else {
        // This Action is being called at least the second time
        // (but not the final time if on node 0).
        reduction_data->at(observation_id)
            .combine(std::move(received_reduction_data));
      }

      // We use a bool `write_to_disk` to allow us to defer the data writing
      // until after we've unlocked the lock. For the same reason, we move the
      // final, reduced result into `received_reduction_data` and
      // `reduction_names`.
      bool write_to_disk = false;
      if (nodes_contributed_to_observation.size() ==
          observations_registered_with_id) {
        write_to_disk = true;
        received_reduction_data =
            std::move(reduction_data->operator[](observation_id));
        reduction_names =
            std::move(reduction_names_map->operator[](observation_id));
        reduction_data->erase(observation_id);
        reduction_names_map->erase(observation_id);
        nodes_contributed->erase(observation_id);
      }
      reduction_data_lock->unlock();

      if (write_to_disk) {
        reduction_file_lock->lock();
        // NOLINTNEXTLINE(bugprone-use-after-move)
        received_reduction_data.finalize();
        WriteReductionData::write_data(
            subfile_name,
            // NOLINTNEXTLINE(bugprone-use-after-move)
            std::move(reduction_names),
            std::move(received_reduction_data.data()),
            Parallel::get<Tags::ReductionFileName>(cache),
            std::make_index_sequence<sizeof...(ReductionDatums)>{});
        reduction_file_lock->unlock();
      }
    } else {
      (void)node_lock;
      (void)observation_id;
      (void)sender_node_number;
      (void)subfile_name;
      (void)reduction_names;
      (void)received_reduction_data;
      ERROR("Could not find one of the tags: "
            << pretty_type::get_name<Tags::ReductionData<ReductionDatums...>>()
            << ' '
            << pretty_type::get_name<
                   Tags::ReductionDataNames<ReductionDatums...>>()
            << ", Tags::NodesThatContributedReductions, "
               "Tags::ReductionDataLock, or Tags::H5FileLock.");
    }
  }
};
}  // namespace ThreadedActions
}  // namespace observers
