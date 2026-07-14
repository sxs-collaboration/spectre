// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace spectre::Exporter {

/*!
 * \brief Reads time series of modal coefficients from volume data files.
 *
 * This class reads tensor components written by multiple observations of a
 * volume data subfile, transforms the nodal data of each element to modal
 * coefficients, and returns the time series of every modal coefficient one
 * element at a time. Reading element-by-element keeps memory usage low: only
 * a single element's time series is held in memory at once, so the volume
 * data never has to be loaded into memory as a whole.
 *
 * Request the time series with `modal_time_series()`, e.g.:
 *
 * \snippet Test_ModalTimeSeriesReader.cpp modal_time_series_reader_example
 *
 * The observation times must be uniformly spaced. Multiple files can be used
 * when the elements of each observation are distributed across files, such as
 * files written by different nodes. All files must contain the same
 * observations in the requested time interval, and each element must reside
 * in the same file with the same mesh for all observations. Files from
 * different simulation segments must be joined first, with overlapping
 * observations removed. Adaptive mesh refinement and element migration
 * between files (e.g. by load balancing) are not supported yet and raise
 * errors.
 *
 * Only the Legendre basis is supported for now.
 *
 * \note When observing fields on a smaller mesh, use the `ProjectToMesh`
 * option of the ObserveFields event to ensure the data is truncated cleanly
 * in modal space. Do not use `InterpolateToMesh` as this creates a
 * catastrophic aliasing error.
 */
template <size_t Dim>
class ModalTimeSeriesReader {
 public:
  /// Modal coefficient time series of a single element, indexed as
  /// `series[component][mode][observation]`. Modes are ordered by the
  /// collapsed index of the element's mesh.
  using Series = std::vector<std::vector<std::vector<double>>>;

  /*!
   * \brief Construct from one or more volume files.
   *
   * Reads and validates metadata from all files. The volume data itself is
   * only read when calling `modal_time_series()`.
   *
   * \param volume_files_or_glob A list of volume H5 files, or a glob string
   *     that resolves to volume files. The files can distribute elements of
   *     the same observations across nodes, but files from different
   *     simulation segments must be joined first.
   * \param subfile_name The name of the volume data subfile in the H5 files.
   * \param tensor_components Tensor component names to read. Each component
   *     must exist in every file at every observation.
   * \param start_time Optional lower bound to restrict observations.
   * \param end_time Optional upper bound to restrict observations.
   */
  ModalTimeSeriesReader(const std::variant<std::vector<std::string>,
                                           std::string>& volume_files_or_glob,
                        std::string subfile_name,
                        std::vector<std::string> tensor_components,
                        std::optional<double> start_time = std::nullopt,
                        std::optional<double> end_time = std::nullopt);

  ModalTimeSeriesReader(ModalTimeSeriesReader&&);
  ModalTimeSeriesReader& operator=(ModalTimeSeriesReader&&);
  ModalTimeSeriesReader(const ModalTimeSeriesReader&) = delete;
  ModalTimeSeriesReader& operator=(const ModalTimeSeriesReader&) = delete;
  ~ModalTimeSeriesReader();

  /// Time of the first observation (after filtering by `start_time` and
  /// `end_time`)
  double start_time() const { return obs_ids_and_times_.front().second; }

  /// Uniform spacing between observation times
  double time_step() const { return time_step_; }

  /// Number of observations (after filtering by `start_time` and `end_time`)
  size_t num_observations() const { return obs_ids_and_times_.size(); }

  /// The tensor components that will be read
  const std::vector<std::string>& tensor_components() const {
    return tensor_components_;
  }

  /*!
   * \brief All elements in the volume files and their meshes.
   *
   * The elements are grouped by the volume file they reside in. Requesting
   * the `modal_time_series()` in this order avoids re-reading per-file
   * metadata.
   */
  const std::vector<std::pair<ElementId<Dim>, Mesh<Dim>>>& elements() const {
    return elements_;
  }

  /*!
   * \brief The time series of all modal coefficients of all tensor
   * components of the given element.
   *
   * Metadata of the file in which the element resides is cached between
   * calls, so requesting elements in the order of `elements()` is most
   * efficient (this is also why this function is not const).
   *
   * Reading the volume data is currently not optimized: every observation of
   * a tensor component is read from disk once per element residing in the
   * file. Reading only the element's subset of the data would avoid this
   * amplification without changing this interface.
   */
  Series modal_time_series(const ElementId<Dim>& element_id);

 private:
  struct FileCache;

  std::vector<std::string> filenames_;
  std::string subfile_name_;
  std::vector<std::string> tensor_components_;
  std::vector<std::pair<size_t, double>> obs_ids_and_times_;
  double time_step_{};
  /// Elements grouped by file, in the order they appear in the files
  std::vector<std::pair<ElementId<Dim>, Mesh<Dim>>> elements_;
  /// Mesh and file index of each element for fast lookup
  std::unordered_map<ElementId<Dim>, std::pair<Mesh<Dim>, size_t>>
      element_info_;
  /// Metadata of the most recently accessed file
  std::unique_ptr<FileCache> file_cache_;
};

}  // namespace spectre::Exporter
