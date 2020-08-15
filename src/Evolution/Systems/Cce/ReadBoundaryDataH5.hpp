// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <memory>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/Cce/BoundaryData.hpp"
#include "Evolution/Systems/Cce/SpecBoundaryData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Interpolation/SpanInterpolator.hpp"
#include "Parallel/CharmPupable.hpp"

namespace Cce {
namespace Tags {
namespace detail {
// tags for use in the buffers for the modal input worldtube data management
// classes
using SpatialMetric =
    gr::Tags::SpatialMetric<3, ::Frame::Inertial, ComplexModalVector>;
using Shift = gr::Tags::Shift<3, ::Frame::Inertial, ComplexModalVector>;
using Lapse = gr::Tags::Lapse<ComplexModalVector>;

// radial derivative prefix tag to be used with the modal input worldtube data
template <typename Tag>
struct Dr : db::SimpleTag, db::PrefixTag {
  using type = typename Tag::type;
  using tag = Tag;
};

// tag for the string for accessing the quantity associated with `Tag` in
// worldtube h5 file
template <typename Tag>
struct InputDataSet : db::SimpleTag, db::PrefixTag {
  using type = std::string;
  using tag = Tag;
};
}  // namespace detail
}  // namespace Tags

namespace detail {
// generates the component dataset name in the worldtube file based on the
// tensor indices requested. For instance, if called with arguments ("/g", 0,1),
// it returns the dataset name "/gxy".
template <typename... T>
std::string dataset_name_for_component(std::string base_name,
                                       const T... indices) noexcept {  // NOLINT
  const auto add_index = [&base_name](size_t index) noexcept {
    ASSERT(index < 3, "The character-arithmetic index must be less than 3.");
    base_name += static_cast<char>('x' + index);
  };
  EXPAND_PACK_LEFT_TO_RIGHT(add_index(indices));
  // void cast so that compilers can tell it's used.
  (void)add_index;
  return base_name;
}

// creates a pair of indices such that the difference is `2 *
// interpolator_length + pad`, centered around `time`, and bounded by
// `lower_bound` and `upper_bound`. If it cannot be centered, it gives a span
// that is appropriately sized and bounded by the supplied bounds. If the bounds
// are too constraining for the necessary size, it gives a span that is the
// correct size starting at `lower bound`, but not constrained by `upper_bound`
std::pair<size_t, size_t> create_span_for_time_value(
    double time, size_t pad, size_t interpolator_length, size_t lower_bound,
    size_t upper_bound, const DataVector& time_buffer) noexcept;
}  // namespace detail

/// the full set of tensors to be extracted from the worldtube h5 file
using cce_input_tags = tmpl::list<
    Tags::detail::SpatialMetric, Tags::detail::Dr<Tags::detail::SpatialMetric>,
    ::Tags::dt<Tags::detail::SpatialMetric>, Tags::detail::Shift,
    Tags::detail::Dr<Tags::detail::Shift>, ::Tags::dt<Tags::detail::Shift>,
    Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
    ::Tags::dt<Tags::detail::Lapse>>;

/// the full set of tensors to be extracted from the reduced form of the
/// worldtube h5 file
using reduced_cce_input_tags =
    tmpl::list<Spectral::Swsh::Tags::SwshTransform<Tags::BondiBeta>,
               Spectral::Swsh::Tags::SwshTransform<Tags::BondiU>,
               Spectral::Swsh::Tags::SwshTransform<Tags::BondiQ>,
               Spectral::Swsh::Tags::SwshTransform<Tags::BondiW>,
               Spectral::Swsh::Tags::SwshTransform<Tags::BondiJ>,
               Spectral::Swsh::Tags::SwshTransform<Tags::Dr<Tags::BondiJ>>,
               Spectral::Swsh::Tags::SwshTransform<Tags::Du<Tags::BondiJ>>,
               Spectral::Swsh::Tags::SwshTransform<Tags::BondiR>,
               Spectral::Swsh::Tags::SwshTransform<Tags::Du<Tags::BondiR>>>;

/// \cond
class MetricWorldtubeH5BufferUpdater;
class BondiWorldtubeH5BufferUpdater;
/// \endcond

/*!
 *  \brief Abstract base class for utilities that are able to perform the buffer
 *  updating procedure needed by the `WorldtubeDataManager`.
 *
 *  \details The methods that are required to be overridden in the derived
 * classes are:
 *  - `WorldtubeBufferUpdater::update_buffers_for_time()`:
 *  updates the buffers passed by pointer and the `time_span_start` and
 *  `time_span_end` to be appropriate for the requested `time`,
 *  `interpolator_length`, and `buffer_depth`.
 *  - `WorldtubeBufferUpdater::get_clone()`
 *  clone function to obtain a `std::unique_ptr` of the base
 *  `WorldtubeBufferUpdater`, needed to pass around the factory-created
 *  object.
 *  - `WorldtubeBufferUpdater::time_is_outside_range()`
 *  the override should return `true` if the `time` could be used in a
 *  `update_buffers_for_time` call given the data available to the derived
 *  class, and `false` otherwise
 *  - `WorldtubeBufferUpdater::get_l_max()`
 *  The override should return the `l_max` it uses in the
 *  Goldberg modal data placed in the buffers.
 *  - `WorldtubeBufferUpdater::get_extraction_radius()`
 *  The override should return the coordinate radius associated with the modal
 *  worldtube data that it supplies in the buffer update function. This is
 *  currently assumed to be a single double, but may be generalized in future
 *  to be time-dependent.
 *  - `WorldtubeBufferUpdater::get_time_buffer`
 *  The override should return the vector of times that it can produce modal
 *  data at. For instance, if associated with a file input, this will be the
 *  times at each of the rows of the time-series data.
 */
template <typename BufferTags>
class WorldtubeBufferUpdater : public PUP::able {
 public:
  using creatable_classes =
      tmpl::list<MetricWorldtubeH5BufferUpdater, BondiWorldtubeH5BufferUpdater>;

  WRAPPED_PUPable_abstract(WorldtubeBufferUpdater);  // NOLINT

  virtual double update_buffers_for_time(
      gsl::not_null<Variables<BufferTags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept = 0;

  virtual std::unique_ptr<WorldtubeBufferUpdater> get_clone()
      const noexcept = 0;

  virtual bool time_is_outside_range(double time) const noexcept = 0;

  virtual size_t get_l_max() const noexcept = 0;

  virtual double get_extraction_radius() const noexcept = 0;

  virtual bool radial_derivatives_need_renormalization() const noexcept = 0;

  virtual DataVector& get_time_buffer() noexcept = 0;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube  H5 file
/// produced by SpEC.
class MetricWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<cce_input_tags> {
 public:
  // charm needs the empty constructor
  MetricWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. Note that this assumes that the input data has
  /// correctly-normalized radial derivatives, and that the extraction radius is
  /// encoded as an integer in the filename.
  explicit MetricWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename) noexcept;

  WRAPPED_PUPable_decl_template(MetricWorldtubeH5BufferUpdater);  // NOLINT

  explicit MetricWorldtubeH5BufferUpdater(
      CkMigrateMessage* /*unused*/) noexcept {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`. The
  /// function returns the next time at which a full update will occur. If
  /// called again at times earlier than the next full update time, it will
  /// leave the `buffers` unchanged and again return the next needed time.
  double update_buffers_for_time(
      gsl::not_null<Variables<cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<WorldtubeBufferUpdater<cce_input_tags>> get_clone()
      const noexcept override;

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(double time) const noexcept override;

  /// retrieves the l_max of the input file
  size_t get_l_max() const noexcept override { return l_max_; }

  /// retrieves the extraction radius encoded in the filename
  double get_extraction_radius() const noexcept override {
    return extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

  bool radial_derivatives_need_renormalization() const noexcept override {
    return radial_derivatives_need_renormalization_;
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  void update_buffer(gsl::not_null<ComplexModalVector*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start,
                     size_t time_span_end) const noexcept;

  bool radial_derivatives_need_renormalization_ = false;
  double extraction_radius_ = 1.0;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, cce_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube H5 file
/// produced by the reduced SpEC format.
class BondiWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<reduced_cce_input_tags> {
 public:
  // charm needs the empty constructor
  BondiWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. Note that this assumes that the input data has
  /// correctly-normalized radial derivatives, and that the extraction radius is
  /// encoded as an integer in the filename.
  explicit BondiWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename) noexcept;

  WRAPPED_PUPable_decl_template(BondiWorldtubeH5BufferUpdater);  // NOLINT

  explicit BondiWorldtubeH5BufferUpdater(
      CkMigrateMessage* /*unused*/) noexcept {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<reduced_cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<WorldtubeBufferUpdater<reduced_cce_input_tags>> get_clone()
      const noexcept override {
    return std::make_unique<BondiWorldtubeH5BufferUpdater>(filename_);
  }

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(const double time) const noexcept override {
    return time < time_buffer_[0] or
           time > time_buffer_[time_buffer_.size() - 1];
  }

  /// retrieves the l_max of the input file
  size_t get_l_max() const noexcept override { return l_max_; }

  /// retrieves the extraction radius encoded in the filename
  double get_extraction_radius() const noexcept override {
    return extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

  bool radial_derivatives_need_renormalization() const noexcept override {
    return false;
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  void update_buffer(gsl::not_null<ComplexModalVector*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start, size_t time_span_end,
                     bool is_real) const noexcept;

  double extraction_radius_ = 1.0;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, reduced_cce_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

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
      double time) const noexcept = 0;

  virtual std::unique_ptr<WorldtubeDataManager> get_clone() const noexcept = 0;

  virtual size_t get_l_max() const noexcept = 0;

  virtual std::pair<size_t, size_t> get_time_span() const noexcept = 0;
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
      std::unique_ptr<WorldtubeBufferUpdater<cce_input_tags>> buffer_updater,
      const size_t l_max, const size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
      : buffer_updater_{std::move(buffer_updater)},
        l_max_{l_max},
        interpolated_coefficients_{
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
        buffer_depth_{buffer_depth},
        interpolator_{std::move(interpolator)} {
    if (UNLIKELY(
            buffer_updater_->get_time_buffer().size() <
            2 * interpolator_->required_number_of_points_before_and_after() +
                buffer_depth)) {
      ERROR(
          "The specified buffer updater doesn't have enough time points to "
          "supply the requested interpolation buffer. This almost certainly "
          "indicates that the corresponding file hasn't been created properly, "
          "but might indicate that the `buffer_depth` template parameter is "
          "too large or the specified Interpolator requests too many points");
    }

    const size_t size_of_buffer =
        square(l_max + 1) *
        (buffer_depth +
         2 * interpolator_->required_number_of_points_before_and_after());
    coefficients_buffers_ = Variables<cce_input_tags>{size_of_buffer};
  }

  WRAPPED_PUPable_decl_template(MetricWorldtubeDataManager);  // NOLINT

  explicit MetricWorldtubeDataManager(CkMigrateMessage* /*unused*/) noexcept {}

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
      double time) const noexcept override;

  std::unique_ptr<WorldtubeDataManager> get_clone() const noexcept override {
    return std::make_unique<MetricWorldtubeDataManager>(
        buffer_updater_->get_clone(), l_max_, buffer_depth_,
        interpolator_->get_clone());
  }

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const noexcept override { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const noexcept override {
    return std::make_pair(time_span_start_, time_span_end_);
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override {  // NOLINT
    p | buffer_updater_;
    p | time_span_start_;
    p | time_span_end_;
    p | l_max_;
    p | buffer_depth_;
    p | interpolator_;
    if (p.isUnpacking()) {
      time_span_start_ = 0;
      time_span_end_ = 0;
      const size_t size_of_buffer =
          square(l_max_ + 1) *
          (buffer_depth_ +
           2 * interpolator_->required_number_of_points_before_and_after());
      coefficients_buffers_ = Variables<cce_input_tags>{size_of_buffer};
      interpolated_coefficients_ = Variables<cce_input_tags>{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
    }
  }

 private:
  std::unique_ptr<WorldtubeBufferUpdater<cce_input_tags>> buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<cce_input_tags> interpolated_coefficients_;

  // note: buffers store data in a 'time-varies-fastest' manner
  mutable Variables<cce_input_tags> coefficients_buffers_;

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
 * `reduced_cce_input_tags`, rather than direct metric components
 * handled by `WorldtubeDataManager`. The set of 9 scalars is a far leaner
 * (factor of ~4) data storage format.
 */
class BondiWorldtubeDataManager : public WorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  BondiWorldtubeDataManager() = default;

  BondiWorldtubeDataManager(
      std::unique_ptr<WorldtubeBufferUpdater<reduced_cce_input_tags>>
          buffer_updater,
      size_t l_max, size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept;

  WRAPPED_PUPable_decl_template(BondiWorldtubeDataManager);  // NOLINT

  explicit BondiWorldtubeDataManager(CkMigrateMessage* /*unused*/) noexcept {}

  /*!
   * \brief Update the `boundary_data_box` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data at
   * `time`.
   *
   * \details First, if the stored buffer requires updating, it will be updated
   * via the `buffer_updater_` supplied in the constructor. Then, each of the
   * 9 spin-weighted scalars in `reduced_cce_input_tags`
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
      double time) const noexcept override;

  std::unique_ptr<WorldtubeDataManager> get_clone() const noexcept override {
    return std::make_unique<BondiWorldtubeDataManager>(
        buffer_updater_->get_clone(), l_max_, buffer_depth_,
        interpolator_->get_clone());
  }

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const noexcept override { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const noexcept override {
    return std::make_pair(time_span_start_, time_span_end_);
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;  // NOLINT

 private:
  std::unique_ptr<WorldtubeBufferUpdater<reduced_cce_input_tags>>
      buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<reduced_cce_input_tags> interpolated_coefficients_;

  // note: buffers store data in an 'time-varies-fastest' manner
  mutable Variables<reduced_cce_input_tags> coefficients_buffers_;

  size_t buffer_depth_ = 0;

  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};
}  // namespace Cce
