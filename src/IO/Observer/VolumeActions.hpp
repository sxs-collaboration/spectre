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
#include "ErrorHandling/Error.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace observers {
/// \cond
namespace ThreadedActions {
struct ContributeVolumeDataToWriter;
}  // namespace ThreadedActions
/// \endcond
namespace Actions {

/*!
 * \ingroup ObserversGroup
 * \brief Send volume tensor data to the observer.
 *
 * The caller of this Action (which is to be invoked on the Observer parallel
 * component) must pass in an `observation_id` used to uniquely identify the
 * observation in time, the name of the `h5::VolumeData` subfile in the HDF5
 * file (e.g. `/element_data`, where the slash is important), the contributing
 * parallel component element's component id, a vector of the `TensorComponent`s
 * to be written to disk, and an `Index<Dim>` of the extents of the volume.
 */
struct ContributeVolumeData {
  template <
      typename ParallelComponent, typename DbTagsList, typename Metavariables,
      typename ArrayIndex, size_t Dim,
      Requires<tmpl::list_contains_v<DbTagsList, Tags::TensorData>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index,
                    const observers::ObservationId& observation_id,
                    const std::string& subfile_name,
                    const observers::ArrayComponentId& sender_array_id,
                    std::vector<TensorComponent>&& received_tensor_data,
                    const Index<Dim>& received_extents,
                    const std::array<Spectral::Basis, Dim>& received_basis,
                    const std::array<Spectral::Quadrature, Dim>&
                        received_quadrature) noexcept {
    db::mutate<Tags::TensorData, Tags::ContributorsOfTensorData>(
        make_not_null(&box),
        [&array_index, &cache, &observation_id, &sender_array_id,
         &received_basis, &received_extents, &received_quadrature,
         &received_tensor_data, &subfile_name](
            const gsl::not_null<std::unordered_map<
                observers::ObservationId,
                std::unordered_map<observers::ArrayComponentId,
                                   ElementVolumeData>>*>
                volume_data,
            const gsl::not_null<std::unordered_map<
                ObservationId, std::unordered_set<ArrayComponentId>>*>
                contributed_volume_data_ids,
            const std::unordered_map<ObservationKey,
                                     std::unordered_set<ArrayComponentId>>&
                registered_array_component_ids) mutable noexcept {
          const ObservationKey& key{observation_id.observation_key()};
          if (UNLIKELY(registered_array_component_ids.find(key) ==
                       registered_array_component_ids.end())) {
            ERROR("Receiving data from observation id "
                  << observation_id << " that was never registered.");
          }
          const auto& registered_ids = registered_array_component_ids.at(key);
          if (UNLIKELY(registered_ids.find(sender_array_id) ==
                       registered_ids.end())) {
            ERROR("Receiving volume data from array component id "
                  << sender_array_id << " that is not registered.");
          }

          auto& contributed_array_ids =
              (*contributed_volume_data_ids)[observation_id];
          if (UNLIKELY(contributed_array_ids.find(sender_array_id) !=
                       contributed_array_ids.end())) {
            ERROR("Already received volume data to observation id "
                  << observation_id << " from array component id "
                  << sender_array_id);
          }
          contributed_array_ids.insert(sender_array_id);

          if (volume_data->count(observation_id) == 0 or
              volume_data->at(observation_id).count(sender_array_id) == 0) {
            std::vector<size_t> extents(received_extents.begin(),
                                        received_extents.end());
            std::vector<Spectral::Basis> bases(received_basis.begin(),
                                               received_basis.end());
            std::vector<Spectral::Quadrature> quadratures(
                received_quadrature.begin(), received_quadrature.end());

            volume_data->operator[](observation_id)
                .emplace(sender_array_id,
                         ElementVolumeData(
                             {received_extents.begin(), received_extents.end()},
                             std::move(received_tensor_data),
                             {received_basis.begin(), received_basis.end()},
                             {received_quadrature.begin(),
                              received_quadrature.end()}));
          } else {
            auto& current_data =
                volume_data->at(observation_id).at(sender_array_id);
            if (UNLIKELY(
                    not alg::equal(current_data.extents, received_extents))) {
              ERROR(
                  "The extents from the same volume component at a specific "
                  "observation should always be the same. For example, the "
                  "extents of a dG element should be the same for all calls to "
                  "ContributeVolumeData that occur at the same time.");
            }
            current_data.tensor_components.insert(
                current_data.tensor_components.end(),
                std::make_move_iterator(received_tensor_data.begin()),
                std::make_move_iterator(received_tensor_data.end()));
          }

          // Check if we have received all "volume" data from the registered
          // elements. If so we copy it to the nodegroup volume writer.
          if (contributed_array_ids.size() == registered_ids.size()) {
            auto& local_writer = *Parallel::get_parallel_component<
                                      ObserverWriter<Metavariables>>(cache)
                                      .ckLocalBranch();
            Parallel::threaded_action<
                ThreadedActions::ContributeVolumeDataToWriter>(
                local_writer, observation_id,
                ArrayComponentId{std::add_pointer_t<ParallelComponent>{nullptr},
                                 Parallel::ArrayIndex<ArrayIndex>(array_index)},
                subfile_name, std::move((*volume_data)[observation_id]));
            volume_data->erase(observation_id);
          }
        },
        db::get<Tags::ObservationsRegistered>(box));
  }
};
}  // namespace Actions

namespace ThreadedActions {
/*!
 * \ingroup ObserversGroup
 * \brief Move data to the observer writer for writing to disk.
 *
 * Once data from all cores is collected this action writes the data to disk.
 */
struct ContributeVolumeDataToWriter {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const gsl::not_null<Parallel::NodeLock*> node_lock,
      const observers::ObservationId& observation_id,
      ArrayComponentId observer_group_id, const std::string& subfile_name,
      std::unordered_map<observers::ArrayComponentId, ElementVolumeData>&&
          received_volume_data) noexcept {
    if constexpr (tmpl::list_contains_v<DbTagsList, Tags::TensorData> and
                  tmpl::list_contains_v<DbTagsList,
                                        Tags::ContributorsOfTensorData> and
                  tmpl::list_contains_v<DbTagsList, Tags::VolumeDataLock> and
                  tmpl::list_contains_v<DbTagsList, Tags::H5FileLock>) {
      // The below gymnastics with pointers is done in order to minimize the
      // time spent locking the entire node, which is necessary because the
      // DataBox does not allow any functions calls, both get and mutate, during
      // a mutate. This design choice in DataBox is necessary to guarantee a
      // consistent state throughout mutation. Here, however, we need to be
      // reasonable efficient in parallel and so we manually guarantee that
      // consistent state. To this end, we create pointers and assign to them
      // the data in the DataBox which is guaranteed to be pointer stable. The
      // data itself is guaranteed to be stable inside the VolumeDataLock.
      std::unordered_map<
          observers::ObservationId,
          std::unordered_map<observers::ArrayComponentId, ElementVolumeData>>*
          all_volume_data = nullptr;
      std::unordered_map<observers::ArrayComponentId, ElementVolumeData>
          volume_data;
      Parallel::NodeLock* volume_file_lock = nullptr;
      std::unordered_map<ObservationId, std::unordered_set<ArrayComponentId>>*
          volume_observers_contributed = nullptr;
      Parallel::NodeLock* volume_data_lock = nullptr;
      size_t observations_registered_with_id =
          std::numeric_limits<size_t>::max();

      node_lock->lock();
      db::mutate<Tags::TensorData, Tags::ContributorsOfTensorData,
                 Tags::VolumeDataLock, Tags::H5FileLock>(
          make_not_null(&box),
          [&observation_id, &observations_registered_with_id,
           &observer_group_id, &all_volume_data, &volume_observers_contributed,
           &volume_data_lock, &volume_file_lock](
              const gsl::not_null<std::unordered_map<
                  observers::ObservationId,
                  std::unordered_map<observers::ArrayComponentId,
                                     ElementVolumeData>>*>
                  volume_data_ptr,
              const gsl::not_null<std::unordered_map<
                  ObservationId, std::unordered_set<ArrayComponentId>>*>
                  volume_observers_contributed_ptr,
              const gsl::not_null<Parallel::NodeLock*> volume_data_lock_ptr,
              const gsl::not_null<Parallel::NodeLock*> volume_file_lock_ptr,
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

            all_volume_data = &*volume_data_ptr;
            volume_observers_contributed = &*volume_observers_contributed_ptr;
            volume_data_lock = &*volume_data_lock_ptr;
            observations_registered_with_id =
                observations_registered.at(key).size();
            volume_file_lock = &*volume_file_lock_ptr;
          },
          db::get<Tags::ObservationsRegistered>(box));
      node_lock->unlock();

      ASSERT(all_volume_data != nullptr,
             "Failed to set all_volume_data in the mutate");
      ASSERT(volume_file_lock != nullptr,
             "Failed to set volume_file_lock in the mutate");
      ASSERT(volume_observers_contributed != nullptr,
             "Failed to set volume_observers_contributed in the mutate");
      ASSERT(volume_data_lock != nullptr,
             "Failed to set volume_data_lock in the mutate");
      ASSERT(
          observations_registered_with_id != std::numeric_limits<size_t>::max(),
          "Failed to set observations_registered_with_id when mutating the "
          "DataBox. This is a bug in the code.");

      volume_data_lock->lock();
      auto& contributed_group_ids =
          (*volume_observers_contributed)[observation_id];

      if (UNLIKELY(contributed_group_ids.find(observer_group_id) !=
                   contributed_group_ids.end())) {
        ERROR("Already received reduction data to observation id "
              << observation_id << " from array component id "
              << observer_group_id);
      }
      contributed_group_ids.insert(observer_group_id);

      if (all_volume_data->find(observation_id) == all_volume_data->end()) {
        // We haven't been called before on this processing element.
        all_volume_data->operator[](observation_id) =
            std::move(received_volume_data);
      } else {
        auto& current_data = all_volume_data->at(observation_id);
        current_data.insert(
            std::make_move_iterator(received_volume_data.begin()),
            std::make_move_iterator(received_volume_data.end()));
      }
      // Check if we have received all "volume" data from the Observer
      // group. If so we write to disk.
      bool perform_write = false;
      if (volume_observers_contributed->at(observation_id).size() ==
          observations_registered_with_id) {
        perform_write = true;
        volume_data = std::move(all_volume_data->operator[](observation_id));
        all_volume_data->erase(observation_id);
        volume_observers_contributed->erase(observation_id);
      }
      volume_data_lock->unlock();

      if (perform_write) {
        ASSERT(not volume_data.empty(),
               "Failed to populate volume_data before trying to write it.");
        // Write to file. We use a separate node lock because writing can be
        // very time consuming (it's network dependent, depends on how full the
        // disks are, what other users are doing, etc.) and we want to be able
        // to continue to work on the nodegroup while we are writing data to
        // disk.
        volume_file_lock->lock();
        {
          // Scoping is for closing HDF5 file before we release the lock.
          const auto& file_prefix = Parallel::get<Tags::VolumeFileName>(cache);
          h5::H5File<h5::AccessType::ReadWrite> h5file(
              file_prefix + std::to_string(Parallel::my_node()) + ".h5", true);
          constexpr size_t version_number = 0;
          auto& volume_file =
              h5file.try_insert<h5::VolumeData>(subfile_name, version_number);
          std::vector<ElementVolumeData> dg_elements;
          dg_elements.reserve(volume_data.size());
          for (const auto& id_and_element : volume_data) {
            dg_elements.push_back(id_and_element.second);
          }
          // Write the data to the file
          volume_file.write_volume_data(observation_id.hash(),
                                        observation_id.value(), dg_elements);
        }
        volume_file_lock->unlock();
      }
    } else {
      (void)node_lock;
      (void)observation_id;
      (void)observer_group_id;
      (void)subfile_name;
      (void)received_volume_data;
      ERROR(
          "Could not find one of the tags TensorData, "
          "ContributorsOfTensorData, "
          "VolumeDataLock, or H5FileLock in the DataBox.");
    }
  }
};
}  // namespace ThreadedActions
}  // namespace observers
