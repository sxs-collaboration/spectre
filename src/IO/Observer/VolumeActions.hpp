// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Domain/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ArrayComponentId.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"
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
 * parallel component element's component id, and the `ElementVolumeData`
 * to be written to disk.
 */
struct ContributeVolumeData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index,
      const observers::ObservationId& observation_id,
      const std::string& subfile_name,
      const Parallel::ArrayComponentId& sender_array_id,
      ElementVolumeData&& received_volume_data,
      const std::optional<std::string>& dependency = std::nullopt) {
    db::mutate<Tags::TensorData, Tags::ContributorsOfTensorData>(
        [&array_index, &cache, &received_volume_data, &observation_id,
         &sender_array_id, &subfile_name, &dependency](
            const gsl::not_null<std::unordered_map<
                observers::ObservationId,
                std::unordered_map<Parallel::ArrayComponentId,
                                   ElementVolumeData>>*>
                volume_data,
            const gsl::not_null<std::unordered_map<
                ObservationId, std::unordered_set<Parallel::ArrayComponentId>>*>
                contributed_volume_data_ids,
            const std::unordered_map<
                ObservationKey, std::unordered_set<Parallel::ArrayComponentId>>&
                registered_array_component_ids) {
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

          if ((not volume_data->contains(observation_id)) or
              (not volume_data->at(observation_id).contains(sender_array_id))) {
            volume_data->operator[](observation_id)
                .emplace(sender_array_id, std::move(received_volume_data));
          } else {
            auto& current_data =
                volume_data->at(observation_id).at(sender_array_id);
            if (UNLIKELY(not alg::equal(current_data.extents,
                                        received_volume_data.extents))) {
              ERROR(
                  "The extents from the same volume component at a specific "
                  "observation should always be the same. For example, the "
                  "extents of a dG element should be the same for all calls to "
                  "ContributeVolumeData that occur at the same time.");
            }
            current_data.tensor_components.insert(
                current_data.tensor_components.end(),
                std::make_move_iterator(
                    received_volume_data.tensor_components.begin()),
                std::make_move_iterator(
                    received_volume_data.tensor_components.end()));
          }

          // Check if we have received all "volume" data from the registered
          // elements. If so we copy it to the nodegroup volume writer.
          if (contributed_array_ids.size() == registered_ids.size()) {
            auto& local_writer = *Parallel::local_branch(
                Parallel::get_parallel_component<ObserverWriter<Metavariables>>(
                    cache));
            Parallel::threaded_action<
                ThreadedActions::ContributeVolumeDataToWriter>(
                local_writer, observation_id,
                Parallel::make_array_component_id<ParallelComponent>(
                    array_index),
                subfile_name, std::move((*volume_data)[observation_id]),
                dependency);
            volume_data->erase(observation_id);
            contributed_volume_data_ids->erase(observation_id);
          }
        },
        make_not_null(&box),
        db::get<Tags::ExpectedContributorsForObservations>(box));
  }
};
}  // namespace Actions

namespace ThreadedActions {
namespace VolumeActions_detail {
void write_data(const std::string& h5_file_name,
                const std::string& input_source,
                const std::string& subfile_path,
                const observers::ObservationId& observation_id,
                std::vector<ElementVolumeData>&& volume_data);

template <typename ParallelComponent, typename Metavariables,
          typename VolumeDataAtObsId>
void write_combined_volume_data(
    Parallel::GlobalCache<Metavariables>& cache,
    const observers::ObservationId& observation_id,
    const VolumeDataAtObsId& volume_data,
    const gsl::not_null<Parallel::NodeLock*> volume_file_lock,
    const std::string& subfile_name) {
  ASSERT(not volume_data.empty(),
         "Failed to populate volume_data before trying to write it.");

  std::vector<ElementVolumeData> volume_data_to_write;

  if constexpr (std::is_same_v<tmpl::at_c<VolumeDataAtObsId, 1>,
                               ElementVolumeData>) {
    volume_data_to_write.reserve(volume_data.size());
    for (const auto& [id, element] : volume_data) {
      (void)id;  // avoid compiler warnings
      volume_data_to_write.push_back(element);
    }
  } else {
    size_t total_size = 0;
    for (const auto& [id, vec_elements] : volume_data) {
      (void)id;  // avoid compiler warnings
      total_size += vec_elements.size();
    }
    volume_data_to_write.reserve(total_size);

    for (const auto& [id, vec_elements] : volume_data) {
      (void)id;  // avoid compiler warnings
      volume_data_to_write.insert(volume_data_to_write.end(),
                                  vec_elements.begin(), vec_elements.end());
    }
  }

  // Write to file. We use a separate node lock because writing can be
  // very time consuming (it's network dependent, depends on how full the
  // disks are, what other users are doing, etc.) and we want to be able
  // to continue to work on the nodegroup while we are writing data to
  // disk.
  const std::lock_guard hold_lock(*volume_file_lock);
  {
    // Scoping is for closing HDF5 file before we release the lock.
    const auto& file_prefix = Parallel::get<Tags::VolumeFileName>(cache);
    auto& my_proxy = Parallel::get_parallel_component<ParallelComponent>(cache);
    h5::H5File<h5::AccessType::ReadWrite> h5file(
        file_prefix +
            std::to_string(
                Parallel::my_node<int>(*Parallel::local_branch(my_proxy))) +
            ".h5",
        true, observers::input_source_from_cache(cache));
    constexpr size_t version_number = 0;
    auto& volume_file =
        h5file.try_insert<h5::VolumeData>(subfile_name, version_number);

    // Serialize domain. See `Domain` docs for details on the serialization.
    // The domain is retrieved from the global cache using the standard
    // domain tag. If more flexibility is required here later, then the
    // domain can be passed along with the `ContributeVolumeData` action.
    std::optional<std::vector<char>> serialized_domain{};
    if (not volume_file.has_domain()) {
      serialized_domain = serialize(
          Parallel::get<domain::Tags::Domain<Metavariables::volume_dim>>(
              cache));
    }

    std::optional<std::vector<char>> serialized_global_functions_of_time =
        std::nullopt;
    std::optional<std::vector<char>> serialized_observation_functions_of_time =
        std::nullopt;
    if constexpr (Parallel::is_in_global_cache<Metavariables,
                                               domain::Tags::FunctionsOfTime>) {
      const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
      serialized_global_functions_of_time = serialize(functions_of_time);
      // NOLINTNEXTLINE(misc-const-correctness)
      domain::FunctionsOfTimeMap observation_functions_of_time{};
      const double obs_time = observation_id.value();
      // Generally, the functions of time should be valid when we
      // perform an observation.  The exception is when running in an
      // AtCleanup event, in which case the observation time is a
      // bogus value and we just skip writing the values.
      if (alg::all_of(functions_of_time, [&](const auto& fot) {
            const auto bounds = fot.second->time_bounds();
            return bounds[0] <= obs_time and obs_time <= bounds[1];
          })) {
        // create a new function of time, effectively truncating the history.
        for (const auto& [name, fot_ptr] : functions_of_time) {
          observation_functions_of_time[name] = fot_ptr->create_at_time(
              obs_time, obs_time + 100.0 *
                                       std::numeric_limits<double>::epsilon() *
                                       std::max(std::abs(obs_time), 1.0));
        }
        serialized_observation_functions_of_time =
            serialize(observation_functions_of_time);
      }
    }

    // Write the data to the file
    volume_file.write_volume_data(observation_id.hash(), observation_id.value(),
                                  volume_data_to_write, serialized_domain,
                                  serialized_observation_functions_of_time,
                                  serialized_global_functions_of_time);
  }
}
}  // namespace VolumeActions_detail
/*!
 * \ingroup ObserversGroup
 * \brief Move data to the observer writer for writing to disk.
 *
 * Once data from all cores is collected this action writes the data to disk if
 * there isn't a dependency. Or if there is a dependency and it has been
 * received already. If there is a dependency but it hasn't been received yet,
 * data will be written by a call to `ContributeDependency`.
 */
struct ContributeVolumeDataToWriter {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const gsl::not_null<Parallel::NodeLock*> node_lock,
      const observers::ObservationId& observation_id,
      Parallel::ArrayComponentId observer_group_id,
      const std::string& subfile_name,
      std::unordered_map<Parallel::ArrayComponentId,
                         std::vector<ElementVolumeData>>&& received_volume_data,
      const std::optional<std::string>& dependency = std::nullopt) {
    apply_impl<Tags::InterpolatorTensorData, ParallelComponent>(
        box, cache, node_lock, observation_id, observer_group_id, subfile_name,
        std::move(received_volume_data), dependency);
  }

  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/,
      const gsl::not_null<Parallel::NodeLock*> node_lock,
      const observers::ObservationId& observation_id,
      Parallel::ArrayComponentId observer_group_id,
      const std::string& subfile_name,
      std::unordered_map<Parallel::ArrayComponentId, ElementVolumeData>&&
          received_volume_data,
      const std::optional<std::string>& dependency = std::nullopt) {
    apply_impl<Tags::TensorData, ParallelComponent>(
        box, cache, node_lock, observation_id, observer_group_id, subfile_name,
        std::move(received_volume_data), dependency);
  }

 private:
  template <typename TensorDataTag, typename ParallelComponent,
            typename DbTagsList, typename Metavariables,
            typename VolumeDataAtObsId>
  static void apply_impl(
      db::DataBox<DbTagsList>& box, Parallel::GlobalCache<Metavariables>& cache,
      const gsl::not_null<Parallel::NodeLock*> node_lock,
      const observers::ObservationId& observation_id,
      Parallel::ArrayComponentId observer_group_id,
      const std::string& subfile_name, VolumeDataAtObsId received_volume_data,
      const std::optional<std::string>& dependency = std::nullopt) {
    // The below gymnastics with pointers is done in order to minimize the
    // time spent locking the entire node, which is necessary because the
    // DataBox does not allow any function calls, either get and mutate, during
    // a mutate. We separate out writing from the operations that edit the
    // DataBox since writing to disk can be very slow, but moving data around is
    // comparatively quick.
    Parallel::NodeLock* volume_file_lock = nullptr;
    bool perform_write = false;
    VolumeDataAtObsId volume_data{};

    {
      const std::lock_guard hold_lock(*node_lock);

      // Set file lock for later
      db::mutate<Tags::H5FileLock>(
          [&volume_file_lock](
              const gsl::not_null<Parallel::NodeLock*> volume_file_lock_ptr) {
            volume_file_lock = &*volume_file_lock_ptr;
          },
          make_not_null(&box));

      ASSERT(volume_file_lock != nullptr,
             "Failed to set volume_file_lock in the mutate");

      const auto& observations_registered =
          db::get<Tags::ExpectedContributorsForObservations>(box);

      const ObservationKey& key = observation_id.observation_key();
      if (LIKELY(observations_registered.contains(key))) {
        if (UNLIKELY(not observations_registered.at(key).contains(
                observer_group_id))) {
          ERROR("The observer group id "
                << observer_group_id
                << " was not registered for the observation id "
                << observation_id);
        }
      } else {
        ERROR("key " << key
                     << " not in the registered group ids. Known keys are "
                     << keys_of(observations_registered));
      }

      const size_t observations_registered_with_id =
          observations_registered.at(key).size();

      // Ok because we have the node lock
      auto& volume_observers_contributed =
          db::get_mutable_reference<Tags::ContributorsOfTensorData>(
              make_not_null(&box));
      auto& all_volume_data =
          db::get_mutable_reference<TensorDataTag>(make_not_null(&box));
      auto& box_dependencies =
          db::get_mutable_reference<Tags::Dependencies>(make_not_null(&box));

      auto& contributed_group_ids =
          volume_observers_contributed[observation_id];

      if (UNLIKELY(contributed_group_ids.contains(observer_group_id))) {
        ERROR("Already received reduction data to observation id "
              << observation_id << " from array component id "
              << observer_group_id);
      }
      contributed_group_ids.insert(observer_group_id);

      // Add received volume data to the box
      if (all_volume_data.contains(observation_id)) {
        auto& current_data = all_volume_data.at(observation_id);
        current_data.insert(
            std::make_move_iterator(received_volume_data.begin()),
            std::make_move_iterator(received_volume_data.end()));
      } else {
        // We haven't been called before on this processing element.
        all_volume_data[observation_id] = std::move(received_volume_data);
      }

      // Check if we have received all "volume" data from the Observer
      // group. If so we write to disk.
      if (volume_observers_contributed.at(observation_id).size() ==
          observations_registered_with_id) {
        // Check if
        //  1. there is an external dependencies
        if (dependency.has_value()) {
          //  2. if there is, that we have received something at this time
          //  3. that the dependencies are the same
          if (box_dependencies.contains(observation_id)) {
            if (UNLIKELY(box_dependencies.at(observation_id).first !=
                         dependency.value())) {
              ERROR(
                  "The dependency that was sent to the ObserverWriter from the "
                  "elements ("
                  << dependency.value()
                  << ") does not match the dependency received from "
                     "ContributeDependency ("
                  << box_dependencies.at(observation_id).first << ").");
            }
            //  4. that we are writing the volume data to disk
            if (box_dependencies.at(observation_id).second) {
              perform_write = true;
              volume_data = std::move(all_volume_data[observation_id]);
            }

            // Whether or not we are writing data to disk, we clean up because
            // we have received both the volume data and the dependency
            all_volume_data.erase(observation_id);
            volume_observers_contributed.erase(observation_id);
            box_dependencies.erase(observation_id);
          }
        } else {
          perform_write = true;
          volume_data = std::move(all_volume_data[observation_id]);
          all_volume_data.erase(observation_id);
          volume_observers_contributed.erase(observation_id);
        }
      }
    }

    if (perform_write) {
      VolumeActions_detail::write_combined_volume_data<ParallelComponent>(
          cache, observation_id, volume_data, make_not_null(volume_file_lock),
          subfile_name);
    }
  }
};

/*!
 * \brief Threaded action that will add a dependency to the ObserverWriter for a
 * given ObservationId ( \p time + \p volume_subfile_name).
 *
 * \details If not all the volume data for this ObservationId has been received
 * yet, then this will just add the dependency to the box and exit without
 * writing anything. If all volume data arrives before this action is called,
 * then it will write out the volume data (or remove it if we aren't writing).
 */
struct ContributeDependency {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const double time, const std::string& dependency,
                    std::string volume_subfile_name,
                    const bool write_volume_data) {
    if (not volume_subfile_name.starts_with("/")) {
      volume_subfile_name = "/" + volume_subfile_name;
    }
    if (not volume_subfile_name.ends_with(".vol")) {
      volume_subfile_name += ".vol";
    }

    const ObservationId observation_id{time, volume_subfile_name};

    // The below gymnastics with pointers is done in order to minimize the
    // time spent locking the entire node, which is necessary because the
    // DataBox does not allow any function calls, either get and mutate, during
    // a mutate. We separate out writing from the operations that edit the
    // DataBox since writing to disk can be very slow, but moving data around is
    // comparatively quick.
    Parallel::NodeLock* volume_file_lock = nullptr;
    bool perform_write = false;
    std::unordered_map<Parallel::ArrayComponentId, ElementVolumeData>
        volume_data{};

    // For now just hold the entire node. We can optimize with different locks
    // later on
    {
      const std::lock_guard hold_lock(*node_lock);

      db::mutate<Tags::H5FileLock, Tags::Dependencies>(
          [&](const gsl::not_null<Parallel::NodeLock*> volume_file_lock_ptr,
              const gsl::not_null<std::unordered_map<
                  ObservationId, std::pair<std::string, bool>>*>
                  dependencies) {
            volume_file_lock = &*volume_file_lock_ptr;
            (*dependencies)[observation_id] =
                std::pair{dependency, write_volume_data};
          },
          make_not_null(&box));

      auto& volume_observers_contributed =
          db::get_mutable_reference<Tags::ContributorsOfTensorData>(
              make_not_null(&box));
      auto& all_volume_data =
          db::get_mutable_reference<Tags::TensorData>(make_not_null(&box));
      auto& box_dependencies =
          db::get_mutable_reference<Tags::Dependencies>(make_not_null(&box));
      const auto& expected_contributors =
          db::get<Tags::ExpectedContributorsForObservations>(box);

      if (not expected_contributors.contains(
              observation_id.observation_key())) {
        ERROR("Key " << observation_id.observation_key()
                     << " was not registered.");
      }

      // We have not received any volume data at this time so we can't do
      // anything
      if (not(volume_observers_contributed.contains(observation_id) and
              all_volume_data.contains(observation_id))) {
        return;
      }

      // Check if we have received all "volume" data from the Observer
      // group. If so we write to disk. Then always delete it since the volume
      // data was waiting for this dependency to arrive to be written
      if (volume_observers_contributed.at(observation_id).size() ==
          expected_contributors.at(observation_id.observation_key()).size()) {
        if (write_volume_data) {
          perform_write = true;
          volume_data = std::move(all_volume_data.at(observation_id));
        }

        all_volume_data.erase(observation_id);
        volume_observers_contributed.erase(observation_id);
        box_dependencies.erase(observation_id);
      }
    }

    if (perform_write) {
      VolumeActions_detail::write_combined_volume_data<ParallelComponent>(
          cache, observation_id, volume_data, make_not_null(volume_file_lock),
          volume_subfile_name);
    }
  }
};

/*!
 * \brief Write volume data (such as surface data) at a given time (specified by
 * an `ObservationId`) without the need to register or reduce anything, e.g.
 * from a singleton component or from a specific chare.
 *
 * Use `observers::Actions::ContributeVolumeDataToWriter` instead if you need to
 * write volume data from an array chare (e.g., writing volume data from all the
 * elements in a domain).
 *
 * Invoke this action on the `observers::ObserverWriter` component on node 0.
 * Pass the following arguments when invoking this action:
 *
 * - `h5_file_name`: the name of the HDF5 file where the volume data is to be
 * written (without the .h5 extension).
 * - `subfile_path`: the path where the volume data should be written in the
 *   HDF5 file. Include a leading slash, e.g., `/AhA`.
 * - `observation_id`: the ObservationId corresponding to the volume data.
 * - `volume_data`: the volume data to be written.
 */
struct WriteVolumeData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, typename... Ts,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& h5_file_name,
                    const std::string& subfile_path,
                    const observers::ObservationId& observation_id,
                    std::vector<ElementVolumeData>&& volume_data) {
    auto& volume_file_lock =
        db::get_mutable_reference<Tags::H5FileLock>(make_not_null(&box));
    const std::lock_guard hold_lock(volume_file_lock);
    VolumeActions_detail::write_data(
        h5_file_name, observers::input_source_from_cache(cache), subfile_path,
        observation_id, std::move(volume_data));
  }

  // For a local synchronous action
  using return_type = void;

  /// \brief The apply call for the local synchronous action
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename... Ts,
            typename DataBox = db::DataBox<DbTagsList>>
  static return_type apply(
      db::DataBox<DbTagsList>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const std::string& h5_file_name, const std::string& subfile_path,
      const observers::ObservationId& observation_id,
      std::vector<ElementVolumeData>&& volume_data) {
    auto& volume_file_lock =
        db::get_mutable_reference<Tags::H5FileLock>(make_not_null(&box));
    const std::lock_guard hold_lock(volume_file_lock);
    VolumeActions_detail::write_data(
        h5_file_name, observers::input_source_from_cache(cache), subfile_path,
        observation_id, std::move(volume_data));
  }
};
}  // namespace ThreadedActions
}  // namespace observers
