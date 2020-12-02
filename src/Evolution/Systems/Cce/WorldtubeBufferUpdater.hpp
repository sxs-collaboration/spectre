// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp17.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
using cce_metric_input_tags = tmpl::list<
    Tags::detail::SpatialMetric, Tags::detail::Dr<Tags::detail::SpatialMetric>,
    ::Tags::dt<Tags::detail::SpatialMetric>, Tags::detail::Shift,
    Tags::detail::Dr<Tags::detail::Shift>, ::Tags::dt<Tags::detail::Shift>,
    Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
    ::Tags::dt<Tags::detail::Lapse>>;

/// the full set of tensors to be extracted from the reduced form of the
/// worldtube h5 file
using cce_bondi_input_tags =
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

  virtual bool has_version_history() const noexcept = 0;

  virtual DataVector& get_time_buffer() noexcept = 0;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube  H5 file
/// produced by SpEC.
class MetricWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<cce_metric_input_tags> {
 public:
  // charm needs the empty constructor
  MetricWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename.
  explicit MetricWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt) noexcept;

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
      gsl::not_null<Variables<cce_metric_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags>> get_clone()
      const noexcept override;

  /// The time can only be supported in the buffer update if it is between the
  /// first and last time of the input file.
  bool time_is_outside_range(double time) const noexcept override;

  /// retrieves the l_max of the input file
  size_t get_l_max() const noexcept override { return l_max_; }

  /// retrieves the extraction radius
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

  bool has_version_history() const noexcept override {
    return has_version_history_;
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  void update_buffer(gsl::not_null<ComplexModalVector*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start,
                     size_t time_span_end) const noexcept;

  bool has_version_history_ = true;
  double extraction_radius_ = std::numeric_limits<double>::signaling_NaN();
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, cce_metric_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube H5 file
/// produced by the reduced SpEC format.
class BondiWorldtubeH5BufferUpdater
    : public WorldtubeBufferUpdater<cce_bondi_input_tags> {
 public:
  // charm needs the empty constructor
  BondiWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. The extraction radius can either be passed in directly,
  /// or if it takes the value `std::nullopt`, then the extraction radius is
  /// retrieved as an integer in the filename.
  explicit BondiWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename,
      std::optional<double> extraction_radius = std::nullopt) noexcept;

  WRAPPED_PUPable_decl_template(BondiWorldtubeH5BufferUpdater);  // NOLINT

  explicit BondiWorldtubeH5BufferUpdater(
      CkMigrateMessage* /*unused*/) noexcept {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<cce_bondi_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<WorldtubeBufferUpdater<cce_bondi_input_tags>> get_clone()
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

  /// retrieves the extraction radius. In most normal circumstances, this will
  /// not be needed for Bondi data.
  double get_extraction_radius() const noexcept override {
    if (not static_cast<bool>(extraction_radius_)) {
      ERROR(
          "Extraction radius has not been set, and was not successfully parsed "
          "from the filename. The extraction radius has been used, so must be "
          "set either by the input file or via the filename.");
    }
    return *extraction_radius_;
  }

  /// The time buffer is supplied by non-const reference to allow views to
  /// easily point into the buffer.
  ///
  /// \warning Altering this buffer outside of the constructor of this class
  /// results in undefined behavior! This should be supplied by const reference
  /// once there is a convenient method of producing a const view of a vector
  /// type.
  DataVector& get_time_buffer() noexcept override { return time_buffer_; }

  bool has_version_history() const noexcept override {
    return true;
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept override;

 private:
  void update_buffer(gsl::not_null<ComplexModalVector*> buffer_to_update,
                     const h5::Dat& read_data, size_t computation_l_max,
                     size_t time_span_start, size_t time_span_end,
                     bool is_real) const noexcept;

  std::optional<double> extraction_radius_ = std::nullopt;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, cce_bondi_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};
}  // namespace Cce
