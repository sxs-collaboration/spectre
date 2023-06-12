// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace observers {
namespace ThreadedActions {

/*!
 * \brief Append data to an h5::Dat subfile of `Tags::VolumeFileName`.
 *
 * \details This is a streamlined interface for getting data to the volume file
 * associated with a node; it will simply write the `.dat` object
 * `subfile_name`, giving it the `file_legend` if it does not yet exist,
 * appending `data_row` to the end of the dat.
 */
struct WriteSimpleData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const std::vector<std::string>& file_legend,
                    const std::vector<double>& data_row,
                    const std::string& subfile_name) {
    Parallel::NodeLock* file_lock = nullptr;
    {
      const std::lock_guard hold_lock(*node_lock);
      db::mutate<Tags::H5FileLock>(
          [&file_lock](const gsl::not_null<Parallel::NodeLock*> in_file_lock) {
            file_lock = in_file_lock;
          },
          make_not_null(&box));
    }

    // scoped to close file
    {
      const std::lock_guard hold_lock(*file_lock);
      const auto& file_prefix = Parallel::get<Tags::VolumeFileName>(cache);
      auto& my_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      h5::H5File<h5::AccessType::ReadWrite> h5file(
          file_prefix +
              std::to_string(
                  Parallel::my_node<int>(*Parallel::local_branch(my_proxy))) +
              ".h5",
          true, observers::input_source_from_cache(cache));
      const size_t version_number = 0;
      auto& output_dataset =
          h5file.try_insert<h5::Dat>(subfile_name, file_legend, version_number);
      output_dataset.append(data_row);
      h5file.close_current_object();
    }
  }
};
}  // namespace ThreadedActions
}  // namespace observers
