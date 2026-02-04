// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/interpolators/cardinal_cubic_b_spline.hpp>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/ElementSearchTree.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace spectre::Exporter {

/*!
 * \brief Interpolate tensor components in both space and time using modal data.
 * This is much more efficient/accurate than interpolating nodal data.
 *
 * \details This class builds a time interpolant for every modal coefficient of
 * every tensor component on each element. It reads volume data, converts nodal
 * values to modal coefficients, and constructs a cardinal cubic B-spline in
 * time for each mode. At evaluation time it locates the element containing the
 * target point, evaluates the modal interpolants at the requested time, and
 * evaluates the mode series at the requested point.
 *
 * How to use: In a simulation, create a few events that write out volume data
 * at different time intervals using ObserveFields. The observation times need
 * to be uniform so this can either be done at a fixed slab interval if the slab
 * size is constant or at fixed times using a dense trigger. The idea is that
 * there is a very coarse grid written very frequently and a finer grid written
 * less frequently. This way, the lowest modes are resolved very accurately and
 * the higher modes are approximated wiht a larger error. A useful choice I
 * found is:
 * - Observe the lowest 3-4 modes very frequently, e.g. every 0.5M or even more
 *    often
 * - Observe the lowest 6-7 modes 10 times less frequently
 * - Observe all modes 10 times less frequently again
 *
 * Important: Use the `ProjectToMesh` option of ObserveFields to ensure that the
 * data is truncated cleanly in modal space. Do not use `InterpolateToMesh` as
 * this creates a catastrophic aliasing error.
 *
 * After that, build the ModalSpacetimeInterpolator and write it to disk. For
 * each tensor component for each element, the interpolator estimates the error
 * of the (0,0,0) mode when using all the observed data. This error is then used
 * to pick a time step for the other modes so that they are interpolated with a
 * conservatively similar error. All modes whose amplitude never exceed the
 * estimated error are dropped entirely as they are not important for
 * reconstructing the field (this can be up to 80% of the modes in my tests).
 * This way, the interpolator only keeps around the data that is necessary and
 * the volume files can be deleted once the spacetime interpolator is built,
 * greatly reducing disk space usage.
 *
 * At the moment, the ModalSpacetimeInterpolator only supports Legendre bases
 * and does not support restarts, AMR or element migration. Since the grid
 * spacing is uniform, it is a good idea to use a separate interpolant for parts
 * of the evolution that are dynamically different (e.g. junk radiation vs
 * inspiral vs merger vs ringdown). In the future, I may add a wrapper that
 * patches together several interpolators.
 *
 * \note Due to a bug in the boost cardinal cubic B-spline implementation that
 * was fixed only recently, see
 * https://github.com/boostorg/math/commit/4809e714d4806c07da3a3def0c4550daa0529b8d,
 * the interpolator can only be built with boost version 1.81 or later.
 */
template <size_t Dim, typename Frame = ::Frame::Inertial>
class ModalSpacetimeInterpolator {
 public:
  ModalSpacetimeInterpolator() = default;

  /*!
   * \brief Construct from one or more volume files.
   *
   * \param volume_files_or_glob A list of volume H5 files or a glob string that
   *     resolves to volume files to read. All files must have identical
   *     observation IDs and observation values. AMR and element migration are
   *     not supported, so each element must live in the same file across all
   *     observations.
   * \param subfiles_coarsest_to_finest Ordered list of volume subfile names,
   *     from the coarsest modal truncation to the finest. Observation times in
   *     each subfile must be uniformly spaced. It is expected (but not
   *     enforced) that the observation time step increases from coarsest to
   *     finest subfile.
   * \param tensor_components Tensor component names to interpolate. Each
   *     component must exist in every subfile listed above.
   * \param start_time Optional start time to restrict observations. Use NaN for
   *     no lower bound. If junk radiation is present in the simulation, it is a
   *     good idea to select a start time after the junk has dissipated.
   * \param end_time Optional end time to restrict observations. Use NaN for no
   *     upper bound.
   * \param verbosity Controls diagnostic output during interpolant build.
   */
  ModalSpacetimeInterpolator(
      std::variant<std::vector<std::string>, std::string> volume_files_or_glob,
      std::vector<std::string> subfiles_coarsest_to_finest,
      std::vector<std::string> tensor_components,
      double start_time = std::numeric_limits<double>::signaling_NaN(),
      double end_time = std::numeric_limits<double>::signaling_NaN(),
      Verbosity verbosity = Verbosity::Quiet);

  /*!
   * \brief Construct from a serialized interpolator H5 file.
   *
   * \param h5_filename Filename of the serialized interpolator.
   * \param group_path H5 group path containing the interpolator data.
   * \param verbosity Controls diagnostic output during load.
   */
  ModalSpacetimeInterpolator(const std::string& h5_filename,
                             const std::string& group_path,
                             Verbosity verbosity = Verbosity::Quiet);

  /*!
   * \brief Serialize the interpolator to an H5 file.
   *
   * \param h5_filename Output filename to create or overwrite.
   * \param group_path H5 group path to store interpolator data.
   */
  void write_to_h5(const std::string& h5_filename,
                   const std::string& group_path) const;

  /*!
   * \brief Interpolate tensor components at a spacetime point.
   *
   * \param result Output buffer sized to the number of tensor components.
   * \param target_point Inertial-space coordinates of the query point.
   * \param time Observation time at which to evaluate.
   * \param block_order Optional block ordering hint to speed up block search.
   */
  void interpolate_to_point(gsl::not_null<std::vector<double>*> result,
                            const tnsr::I<double, Dim, Frame>& target_point,
                            double time,
                            std::optional<gsl::not_null<std::vector<size_t>*>>
                                block_order = std::nullopt) const;

  /*!
   * \brief Access the tensor component names.
   */
  const std::vector<std::string>& tensor_components() const {
    return tensor_components_;
  }
  /*!
   * \brief Access the time bounds.
   */
  const std::array<double, 2>& time_bounds() const { return time_bounds_; }

 private:
  struct ModeInterpolator {
    std::optional<boost::math::interpolators::cardinal_cubic_b_spline<double>>
        interpolant{};
    double start_time{0.0};
    double time_step{0.0};
    std::vector<double> values{};
  };

  struct ComponentInterpolator {
    std::vector<ModeInterpolator> modal_interpolants{};
  };

  struct ElementData {
    Mesh<Dim> mesh{};
    size_t file_index{};
    std::vector<ComponentInterpolator> component_interpolators{};
  };

  struct FileCache;

  void gather_element_metadata(const std::vector<std::string>& filenames,
                               const std::string& subfile_name,
                               size_t reference_obs_id);
  void build_interpolators(const std::vector<std::string>& filenames,
                           const std::vector<std::string>& subfile_names,
                           double start_time, double end_time);
  std::pair<std::vector<std::vector<double>>, Index<Dim>>
  load_component_time_series(
      const FileCache& file_cache, const ElementId<Dim>& element_id,
      size_t component_index,
      const std::vector<std::pair<size_t, double>>& obs_ids_and_times) const;

  std::variant<std::vector<std::string>, std::string> volume_files_or_glob_;
  std::vector<std::string> tensor_components_;

  std::array<double, 2> time_bounds_{{-std::numeric_limits<double>::infinity(),
                                      std::numeric_limits<double>::infinity()}};

  Verbosity verbosity_{Verbosity::Quiet};
  Domain<Dim> domain_{};
  domain::FunctionsOfTimeMap functions_of_time_{};
  std::map<size_t, domain::ElementSearchTree<Dim>> element_search_trees_;
  std::unordered_map<ElementId<Dim>, ElementData> element_data_;
};

}  // namespace spectre::Exporter
