// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

/// \cond
class MetricWorldtubeDataManager;
class BondiWorldtubeDataManager;
/// \endcond

/*!
 *  \brief Abstract base class for managers of CCE worldtube data that is
 * provided in large time-series chunks, especially the type provided by input
 * h5 files.
 *
 *  \details The methods that are required to be overridden in the derived
 * classes are:
 *
 * - `WorldtubeDataManager::populate_hypersurface_boundary_data()`:
 *   updates the Variables passed by pointer to contain correct boundary data
 *   for the time value passed in. This function should update all of the tags
 *   in `Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>`.
 * - `WorldtubeDataManager::get_clone()`: clone function to obtain a
 *   `std::unique_ptr` of the base `WorldtubeDataManager`, needed to pass around
 *   the factory-created object.
 * - `WorldtubeDataManager::get_l_max()`: The override should return the
 *   `l_max` that it computes for the collocation data calculated during
 *   `WorldtubeDataManager::populate_hypersurface_boundary_data()`.
 * - `WorldtubeBufferUpdater::get_time_span()`: The override should return the
 *   `std::pair` of indices that represent the start and end point of the
 *   underlying data source. This is primarily used for monitoring the frequency
 *   and size of the buffer updates.
 */
class WorldtubeDataManager : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<MetricWorldtubeDataManager, BondiWorldtubeDataManager>;

  WRAPPED_PUPable_abstract(WorldtubeDataManager);  // NOLINT

  virtual bool populate_hypersurface_boundary_data(
      gsl::not_null<Variables<
          Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time, gsl::not_null<Parallel::NodeLock*> hdf5_lock) const = 0;

  virtual std::unique_ptr<WorldtubeDataManager> get_clone() const = 0;

  virtual size_t get_l_max() const = 0;

  virtual std::pair<size_t, size_t> get_time_span() const = 0;
};

/*!
 * \brief Manages the cached buffer data associated with a CCE worldtube and
 * interpolates to requested time points to provide worldtube boundary data to
 * the main evolution routines.
 *
 * \details The maintained buffer will be maintained at a length that is set by
 * the `Interpolator` and the `buffer_depth` also passed to the constructor. A
 * longer depth will ensure that the buffer updater is called less frequently,
 * which is useful for slow updaters (e.g. those that perform file access).
 * The main functionality is provided by the
 * `WorldtubeDataManager::populate_hypersurface_boundary_data()` member
 * function that handles buffer updating and boundary computation.
 */
class MetricWorldtubeDataManager : public WorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  MetricWorldtubeDataManager() = default;

  MetricWorldtubeDataManager(
      std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags>>
          buffer_updater,
      size_t l_max, size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator,
      bool fix_spec_normalization);

  WRAPPED_PUPable_decl_template(MetricWorldtubeDataManager);  // NOLINT

  explicit MetricWorldtubeDataManager(CkMigrateMessage* /*unused*/) {}

  /*!
   * \brief Update the `boundary_data_box` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data at
   * `time`.
   *
   * \details First, if the stored buffer requires updating, it will be updated
   * via the `buffer_updater_` supplied in the constructor. Then, each of the
   * spatial metric, shift, lapse, and each of their radial and time derivatives
   * are interpolated across buffer points to the requested time value (via the
   * `Interpolator` provided in the constructor). Finally, that data is supplied
   * to the `create_bondi_boundary_data()`, which updates the
   * `boundary_data_box` with the Bondi spin-weighted scalars determined from
   * the interpolated Cartesian data.
   *
   * Returns `true` if the time can be supplied from the `buffer_updater_`, and
   * `false` otherwise. No tags are updated if `false` is returned.
   */
  bool populate_hypersurface_boundary_data(
      gsl::not_null<Variables<
          Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time, gsl::not_null<Parallel::NodeLock*> hdf5_lock) const override;

  std::unique_ptr<WorldtubeDataManager> get_clone() const override;

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const override;

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;  // NOLINT

 private:
  std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags>>
      buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;
  bool fix_spec_normalization_ = false;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<cce_metric_input_tags> interpolated_coefficients_;

  // note: buffers store data in a 'time-varies-fastest' manner
  mutable Variables<cce_metric_input_tags> coefficients_buffers_;

  size_t buffer_depth_ = 0;

  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};

/*!
 * \brief Manages the 'reduced' cached buffer dataset associated with a CCE
 * worldtube and interpolates to requested time points to provide worldtube
 * boundary data to the main evolution routines.
 *
 * \details The maintained buffer will be kept at a length that is set by
 * the `Interpolator` and the `buffer_depth` also passed to the constructor. A
 * longer depth will ensure that the buffer updater is called less frequently,
 * which is useful for slow updaters (e.g. those that perform file access).
 * The main functionality is provided by the
 * `WorldtubeDataManager::populate_hypersurface_boundary_data()` member
 * function that handles buffer updating and boundary computation. This version
 * of the data manager handles the 9 scalars of
 * `cce_bondi_input_tags`, rather than direct metric components
 * handled by `WorldtubeDataManager`. The set of 9 scalars is a far leaner
 * (factor of ~4) data storage format.
 */
class BondiWorldtubeDataManager : public WorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  BondiWorldtubeDataManager() = default;

  BondiWorldtubeDataManager(
      std::unique_ptr<WorldtubeBufferUpdater<cce_bondi_input_tags>>
          buffer_updater,
      size_t l_max, size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator);

  WRAPPED_PUPable_decl_template(BondiWorldtubeDataManager);  // NOLINT

  explicit BondiWorldtubeDataManager(CkMigrateMessage* /*unused*/) {}

  /*!
   * \brief Update the `boundary_data_box` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data at
   * `time`.
   *
   * \details First, if the stored buffer requires updating, it will be updated
   * via the `buffer_updater_` supplied in the constructor. Then, each of the
   * 9 spin-weighted scalars in `cce_bondi_input_tags`
   * are interpolated across buffer points to the requested time value (via the
   * `Interpolator` provided in the constructor). Finally, the remaining two
   * scalars not directly supplied in the input file are calculated in-line and
   * put in the \ref DataBoxGroup.
   *
   * Returns `true` if the time can be supplied from the `buffer_updater_`, and
   * `false` otherwise. No tags are updated if `false` is returned.
   */
  bool populate_hypersurface_boundary_data(
      gsl::not_null<Variables<
          Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
          boundary_data_variables,
      double time, gsl::not_null<Parallel::NodeLock*> hdf5_lock) const override;

  std::unique_ptr<WorldtubeDataManager> get_clone() const override;

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const override { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const override;

  /// Serialization for Charm++.
  void pup(PUP::er& p) override;  // NOLINT

 private:
  std::unique_ptr<WorldtubeBufferUpdater<cce_bondi_input_tags>> buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<cce_bondi_input_tags> interpolated_coefficients_;

  // note: buffers store data in an 'time-varies-fastest' manner
  mutable Variables<cce_bondi_input_tags> coefficients_buffers_;

  size_t buffer_depth_ = 0;

  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};
}  // namespace Cce
