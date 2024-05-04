// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Cce.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/Tags.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace Cce::Actions {
/*!
 * \brief Write a single row of data into an `h5::Cce` subfile within an H5
 * file.
 *
 * \details This action can be called as either a threaded action or a local
 * synchronous action. In both cases, invoke this action on the
 * `observers::ObserverWriter` component on node 0. You must pass the following
 * arguments when invoking this action
 *
 * - `subfile_name`: the name of the `h5::Cce` subfile in the HDF5 file.
 * - `l_max`: the number of spherical harmonics of the data
 * - `data`: a `std::unordered_map<std::string, std::vector<double>>` where the
 *   keys are the names of all the bondi quantities to write and the data are
 *   the spherical harmonic coefficients. See `h5::Cce` for exactly what names
 *   need to be used and the layout of the data.
 *
 * \note If you want to write data into an `h5::Dat` file, use
 * `observers::ThreadedActions::WriteReductionDataRow`.
 */
struct WriteScriBondiQuantities {
  /// \brief The apply call for the threaded action
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> node_lock,
                    const std::string& subfile_name, const size_t l_max,
                    std::unordered_map<std::string, std::vector<double>> data) {
    apply<ParallelComponent>(box, node_lock, cache, subfile_name, l_max,
                             std::move(data));
  }

  // The local synchronous action
  using return_type = void;

  /// \brief The apply call for the local synchronous action
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static return_type apply(
      db::DataBox<DbTagList>& box,
      const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const std::string& subfile_name, const size_t l_max,
      std::unordered_map<std::string, std::vector<double>> data) {
    auto& reduction_file_lock =
        db::get_mutable_reference<observers::Tags::H5FileLock>(
            make_not_null(&box));
    const std::lock_guard hold_lock(reduction_file_lock);

    // Make sure all data is the proper size
    ASSERT(
        alg::all_of(data,
                    [&l_max](const auto& name_and_data) {
                      return name_and_data.second.size() ==
                             2 * square(l_max + 1) + 1;
                    }),
        "Some data sent to WriteScriBondiQuantities is not of the proper size "
            << 2 * square(l_max + 1) + 1);

    const std::string input_source = observers::input_source_from_cache(cache);
    const std::string& reduction_file_prefix =
        Parallel::get<observers::Tags::ReductionFileName>(cache);
    h5::H5File<h5::AccessType::ReadWrite> h5file(reduction_file_prefix + ".h5",
                                                 true, input_source);
    constexpr size_t version_number = 0;
    auto& cce_file =
        h5file.try_insert<h5::Cce>(subfile_name, l_max, version_number);
    cce_file.append(data);
  }
};
}  // namespace Cce::Actions
