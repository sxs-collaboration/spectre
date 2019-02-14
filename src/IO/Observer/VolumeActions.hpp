// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <iterator>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
namespace ThreadedActions {
/// \cond
struct WriteVolumeData;
/// \endcond
}  // namespace ThreadedActions

namespace Actions {
/// \cond
struct ContributeVolumeDataToWriter;
/// \endcond

/*!
 * \ingroup ObserverGroup
 * \brief Send volume tensor data to the observer.
 *
 * The caller of this Action (which is to be invoked on the Observer parallel
 * component) must pass in an `observation_id` used to uniquely identify the
 * observation in time, the name of the `h5::VolumeData` subfile in the HDF5
 * file (e.g. `/element_data`, where the slash is important), the contributing
 * parallel component element's component id, a vector of the `TensorComponent`s
 * to be written to disk, and an `Index<Dim>` of the extents of the volume.
 *
 * \warning  Currently this action can only be called once per `observation_id`
 * per `array_component_id`. Calling it more often is undefined behavior and may
 * result in incomplete data being written and/or memory leaks. The reason is
 * that we do not register the number of times a component with an ID at a
 * specific observation id will call the observer. However, this will be a
 * feature implemented in the future.
 */
struct ContributeVolumeData {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent, size_t Dim,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& array_component_id,
                    std::vector<TensorComponent>&& in_received_tensor_data,
                    const Index<Dim>& received_extents) noexcept {
    db::mutate<Tags::TensorData>(
        make_not_null(&box),
        [
          &cache, &observation_id, &array_component_id, &received_extents,
          received_tensor_data = std::move(in_received_tensor_data), &
          subfile_name
        ](const gsl::not_null<db::item_type<Tags::TensorData>*> volume_data,
          const std::unordered_set<ArrayComponentId>&
              volume_component_ids) mutable noexcept {
          if (volume_data->count(observation_id) == 0 or
              volume_data->at(observation_id).count(array_component_id) == 0) {
            std::vector<size_t> extents(received_extents.begin(),
                                        received_extents.end());
            volume_data->operator[](observation_id)
                .emplace(
                    array_component_id,
                    ExtentsAndTensorVolumeData(
                        std::move(extents), std::move(received_tensor_data)));
          } else {
            auto& current_data =
                volume_data->at(observation_id).at(array_component_id);
            ASSERT(
                alg::equal(current_data.extents, received_extents),
                "The extents from the same volume component at a specific "
                "observation should always be the same. For example, the "
                "extents of a dG element should be the same for all calls to "
                "ContributeVolumeData that occur at the same time.");
            current_data.tensor_components.insert(
                current_data.tensor_components.end(),
                std::make_move_iterator(received_tensor_data.begin()),
                std::make_move_iterator(received_tensor_data.end()));
          }

          // Check if we have received all "volume" data from the registered
          // elements. If so we copy it to the nodegroup volume writer.
          if (volume_data->at(observation_id).size() ==
              volume_component_ids.size()) {
            auto& local_writer = *Parallel::get_parallel_component<
                                      ObserverWriter<Metavariables>>(cache)
                                      .ckLocalBranch();
            Parallel::simple_action<ContributeVolumeDataToWriter>(
                local_writer, observation_id, subfile_name,
                std::move((*volume_data)[observation_id]));
            volume_data->erase(observation_id);
          }
        },
        db::get<Tags::VolumeArrayComponentIds>(box));
  }
};

/*!
 * \ingroup ObserverGroup
 * \brief Move data to the observer writer for writing to disk.
 */
struct ContributeVolumeDataToWriter {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static void apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    std::unordered_map<observers::ArrayComponentId,
                                       ExtentsAndTensorVolumeData>&&
                        in_volume_data) noexcept {
    // This is the number of callers that have registered (that are associated
    // with the observation type of this observation_id).
    // We expect that this Action will be called once by each of them.
    const auto expected_number_of_calls = [&box, &observation_id]() noexcept {
      const auto hash = observation_id.observation_type_hash();
      const auto& registered = db::get<Tags::VolumeObserversRegistered>(box);
      return (registered.count(hash) == 1) ? registered.at(hash).size() : 0;
    }();
    db::mutate<Tags::TensorData, Tags::VolumeObserversContributed>(
        make_not_null(&box),
        [
          &cache, &expected_number_of_calls, &observation_id,
          in_volume_data = std::move(in_volume_data), &subfile_name
        ](const gsl::not_null<std::unordered_map<
              observers::ObservationId,
              std::unordered_map<observers::ArrayComponentId,
                                 ExtentsAndTensorVolumeData>>*>
              volume_data,
          const gsl::not_null<
              std::unordered_map<observers::ObservationId, size_t>*>
              volume_observers_contributed) mutable noexcept {
          if (volume_data->count(observation_id) == 0) {
            // We haven't been called before on this processing element.
            volume_data->operator[](observation_id) = std::move(in_volume_data);
            (*volume_observers_contributed)[observation_id] = 1;
          } else {
            auto& current_data = volume_data->at(observation_id);
            current_data.insert(std::make_move_iterator(in_volume_data.begin()),
                                std::make_move_iterator(in_volume_data.end()));
            (*volume_observers_contributed)[observation_id]++;
          }
          // Check if we have received all "volume" data from the Observer
          // group. If so we write to disk.
          if (volume_observers_contributed->at(observation_id) ==
              expected_number_of_calls) {
            Parallel::threaded_action<ThreadedActions::WriteVolumeData>(
                Parallel::get_parallel_component<ObserverWriter<Metavariables>>(
                    cache)[static_cast<size_t>(Parallel::my_node())],
                observation_id, subfile_name);
            volume_observers_contributed->erase(observation_id);
          }
        });
  }
};
}  // namespace Actions

namespace ThreadedActions {
/*!
 * \ingroup ObserverGroup
 * \brief Writes volume data at the `observation_id` to disk.
 */
struct WriteVolumeData {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent,
            Requires<sizeof...(DbTags) != 0> = nullptr>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    const gsl::not_null<CmiNodeLock*> node_lock,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name) noexcept {
    // Get data from the DataBox in a thread-safe manner
    Parallel::lock(node_lock);
    std::unordered_map<observers::ArrayComponentId, ExtentsAndTensorVolumeData>
        volume_data{};
    CmiNodeLock file_lock;
    db::mutate<Tags::H5FileLock, Tags::TensorData>(
        make_not_null(&box),
        [&observation_id, &file_lock, &volume_data ](
            const gsl::not_null<CmiNodeLock*> in_file_lock,
            const gsl::not_null<db::item_type<Tags::TensorData>*>
                in_volume_data) noexcept {
          volume_data = std::move((*in_volume_data)[observation_id]);
          in_volume_data->erase(observation_id);
          file_lock = *in_file_lock;
        });
    Parallel::unlock(node_lock);

    // Write to file. We use a separate node lock because writing can be very
    // time consuming (it's network dependent, depends on how full the disks
    // are, what other users are doing, etc.) and we want to be able to continue
    // to work on the nodegroup while we are writing data to disk.
    Parallel::lock(&file_lock);
    {
      // Scoping is for closing HDF5 file before we release the lock.
      const auto& file_prefix =
          Parallel::get<OptionTags::VolumeFileName>(cache);
      h5::H5File<h5::AccessType::ReadWrite> h5file(
          file_prefix + std::to_string(Parallel::my_node()) + ".h5", true);
      constexpr size_t version_number = 0;
      auto& volume_file =
          h5file.try_insert<h5::VolumeData>(subfile_name, version_number);
      for (const auto& id_and_tensor_data_for_grid : volume_data) {
        const auto& extents_and_tensors = id_and_tensor_data_for_grid.second;
        volume_file.insert_tensor_data(
            observation_id.hash(), observation_id.value(), extents_and_tensors);
      }
    }
    Parallel::unlock(&file_lock);
  }
};
}  // namespace ThreadedActions
}  // namespace observers
