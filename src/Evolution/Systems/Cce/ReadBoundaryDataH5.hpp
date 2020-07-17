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

// the full set of tensors to be extracted from the worldtube h5 file
using cce_input_tags = tmpl::list<
    Tags::detail::SpatialMetric, Tags::detail::Dr<Tags::detail::SpatialMetric>,
    ::Tags::dt<Tags::detail::SpatialMetric>, Tags::detail::Shift,
    Tags::detail::Dr<Tags::detail::Shift>, ::Tags::dt<Tags::detail::Shift>,
    Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
    ::Tags::dt<Tags::detail::Lapse>>;

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

/// \cond
class SpecWorldtubeH5BufferUpdater;
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
class WorldtubeBufferUpdater : public PUP::able {
 public:
  using creatable_classes = tmpl::list<SpecWorldtubeH5BufferUpdater>;

  WRAPPED_PUPable_abstract(WorldtubeBufferUpdater);  // NOLINT

  virtual double update_buffers_for_time(
      gsl::not_null<Variables<detail::cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept = 0;

  virtual std::unique_ptr<WorldtubeBufferUpdater> get_clone() const
      noexcept = 0;

  virtual bool time_is_outside_range(double time) const noexcept = 0;

  virtual size_t get_l_max() const noexcept = 0;

  virtual double get_extraction_radius() const noexcept = 0;

  virtual bool radial_derivatives_need_renormalization() const noexcept = 0;

  virtual DataVector& get_time_buffer() noexcept = 0;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube  H5 file
/// produced by SpEC.
class SpecWorldtubeH5BufferUpdater : public WorldtubeBufferUpdater {
 public:
  // charm needs the empty constructor
  SpecWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. Note that this assumes that the input data has
  /// correctly-normalized radial derivatives, and that the extraction radius is
  /// encoded as an integer in the filename.
  explicit SpecWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename) noexcept;

  WRAPPED_PUPable_decl_template(SpecWorldtubeH5BufferUpdater);  // NOLINT

  explicit SpecWorldtubeH5BufferUpdater(CkMigrateMessage* /*unused*/) noexcept {
  }

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`. The
  /// function returns the next time at which a full update will occur. If
  /// called again at times earlier than the next full update time, it will
  /// leave the `buffers` unchanged and again return the next needed time.
  double update_buffers_for_time(
      gsl::not_null<Variables<detail::cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<WorldtubeBufferUpdater> get_clone() const noexcept override;

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
                     size_t time_span_start, size_t time_span_end) const
      noexcept;

  bool radial_derivatives_need_renormalization_ = false;
  double extraction_radius_ = 1.0;
  size_t l_max_ = 0;

  h5::H5File<h5::AccessType::ReadOnly> cce_data_file_;
  std::string filename_;

  tuples::tagged_tuple_from_typelist<
      db::wrap_tags_in<Tags::detail::InputDataSet, detail::cce_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
};

/// \cond
class ReducedSpecWorldtubeH5BufferUpdater;
/// \endcond

class ReducedWorldtubeBufferUpdater : public PUP::able {
 public:
  using creatable_classes = tmpl::list<ReducedSpecWorldtubeH5BufferUpdater>;

  WRAPPED_PUPable_abstract(ReducedWorldtubeBufferUpdater);  // NOLINT

  virtual double update_buffers_for_time(
      gsl::not_null<Variables<detail::reduced_cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept = 0;

  virtual std::unique_ptr<ReducedWorldtubeBufferUpdater> get_clone() const
      noexcept = 0;

  virtual bool time_is_outside_range(double time) const noexcept = 0;

  virtual size_t get_l_max() const noexcept = 0;

  virtual double get_extraction_radius() const noexcept = 0;

  virtual DataVector& get_time_buffer() noexcept = 0;
};

/// A `WorldtubeBufferUpdater` specialized to the CCE input worldtube H5 file
/// produced by the reduced SpEC format.
class ReducedSpecWorldtubeH5BufferUpdater
    : public ReducedWorldtubeBufferUpdater {
 public:
  // charm needs the empty constructor
  ReducedSpecWorldtubeH5BufferUpdater() = default;

  /// The constructor takes the filename of the SpEC h5 file that will be used
  /// for boundary data. Note that this assumes that the input data has
  /// correctly-normalized radial derivatives, and that the extraction radius is
  /// encoded as an integer in the filename.
  explicit ReducedSpecWorldtubeH5BufferUpdater(
      const std::string& cce_data_filename) noexcept;

  WRAPPED_PUPable_decl_template(ReducedSpecWorldtubeH5BufferUpdater);  // NOLINT

  explicit ReducedSpecWorldtubeH5BufferUpdater(
      CkMigrateMessage* /*unused*/) noexcept {}

  /// update the `buffers`, `time_span_start`, and `time_span_end` with
  /// time-varies-fastest, Goldberg modal data and the start and end index in
  /// the member `time_buffer_` covered by the newly updated `buffers`.
  double update_buffers_for_time(
      gsl::not_null<Variables<detail::reduced_cce_input_tags>*> buffers,
      gsl::not_null<size_t*> time_span_start,
      gsl::not_null<size_t*> time_span_end, double time,
      size_t computation_l_max, size_t interpolator_length,
      size_t buffer_depth) const noexcept override;

  std::unique_ptr<ReducedWorldtubeBufferUpdater> get_clone() const
      noexcept override {
    return std::make_unique<ReducedSpecWorldtubeH5BufferUpdater>(filename_);
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

  tuples::tagged_tuple_from_typelist<db::wrap_tags_in<
      Tags::detail::InputDataSet, detail::reduced_cce_input_tags>>
      dataset_names_;

  // stores all the times in the input file
  DataVector time_buffer_;
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
class WorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  WorldtubeDataManager() noexcept = default;

  WorldtubeDataManager(
      std::unique_ptr<WorldtubeBufferUpdater> buffer_updater,
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
    coefficients_buffers_ = Variables<detail::cce_input_tags>{size_of_buffer};
  }

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
  template <typename TagList>
  bool populate_hypersurface_boundary_data(
      const gsl::not_null<db::DataBox<TagList>*> boundary_data_box,
      const double time) const noexcept {
    if (buffer_updater_->time_is_outside_range(time)) {
      return false;
    }
    buffer_updater_->update_buffers_for_time(
        make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
        make_not_null(&time_span_end_), time, l_max_,
        interpolator_->required_number_of_points_before_and_after(),
        buffer_depth_);
    const auto interpolation_time_span = detail::create_span_for_time_value(
        time, 0, interpolator_->required_number_of_points_before_and_after(),
        time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

    // search through and find the two interpolation points the time point is
    // between. If we can, put the range for the interpolation centered on the
    // desired point. If that can't be done (near the start or the end of the
    // simulation), make the range terminated at the start or end of the cached
    // data and extending for the desired range in the other direction.
    const size_t buffer_span_size = time_span_end_ - time_span_start_;
    const size_t interpolation_span_size =
        interpolation_time_span.second - interpolation_time_span.first;

    const DataVector time_points{buffer_updater_->get_time_buffer().data() +
                                     interpolation_time_span.first,
                                 interpolation_span_size};

    auto interpolate_from_column = [
      &time, &time_points, &buffer_span_size, &interpolation_time_span,
      &interpolation_span_size,
      this
    ](auto data, const size_t column) noexcept {
      auto interp_val = interpolator_->interpolate(
          gsl::span<const double>(time_points.data(), time_points.size()),
          gsl::span<const std::complex<double>>(
              data + column * buffer_span_size +
                  (interpolation_time_span.first - time_span_start_),
              interpolation_span_size),
          time);
      return interp_val;
    };

    // the ComplexModalVectors should be provided from the buffer_updater_ in
    // 'Goldberg' format, so we iterate over modes and convert to libsharp
    // format.

    // we'll just use this buffer to reference into the actual data to satisfy
    // the swsh interface requirement that the spin-weight be labelled with
    // `SpinWeighted`
    SpinWeighted<ComplexModalVector, 0> spin_weighted_buffer;
    for (const auto& libsharp_mode :
         Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          tmpl::for_each<
              tmpl::list<Tags::detail::SpatialMetric,
                         Tags::detail::Dr<Tags::detail::SpatialMetric>,
                         ::Tags::dt<Tags::detail::SpatialMetric>>>([
            this, &i, &j, &libsharp_mode, &interpolate_from_column, &
            spin_weighted_buffer
          ](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            spin_weighted_buffer.set_data_ref(
                get<tag>(interpolated_coefficients_).get(i, j).data(),
                Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
            Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
                libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i, j).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        static_cast<int>(libsharp_mode.m))),
                interpolate_from_column(
                    get<tag>(coefficients_buffers_).get(i, j).data(),
                    Spectral::Swsh::goldberg_mode_index(
                        l_max_, libsharp_mode.l,
                        -static_cast<int>(libsharp_mode.m))));
          });
        }
        tmpl::for_each<tmpl::list<Tags::detail::Shift,
                                  Tags::detail::Dr<Tags::detail::Shift>,
                                  ::Tags::dt<Tags::detail::Shift>>>([
          this, &i, &libsharp_mode, &interpolate_from_column, &
          spin_weighted_buffer
        ](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          spin_weighted_buffer.set_data_ref(
              get<tag>(interpolated_coefficients_).get(i).data(),
              Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
          Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
              libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
              interpolate_from_column(
                  get<tag>(coefficients_buffers_).get(i).data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      static_cast<int>(libsharp_mode.m))),
              interpolate_from_column(
                  get<tag>(coefficients_buffers_).get(i).data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      -static_cast<int>(libsharp_mode.m))));
        });
      }
      tmpl::for_each<
          tmpl::list<Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
                     ::Tags::dt<Tags::detail::Lapse>>>([
        this, &libsharp_mode, &interpolate_from_column, &spin_weighted_buffer
      ](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        spin_weighted_buffer.set_data_ref(
            get(get<tag>(interpolated_coefficients_)).data(),
            Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
        Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
            libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
            interpolate_from_column(get(get<tag>(coefficients_buffers_)).data(),
                                    Spectral::Swsh::goldberg_mode_index(
                                        l_max_, libsharp_mode.l,
                                        static_cast<int>(libsharp_mode.m))),
            interpolate_from_column(get(get<tag>(coefficients_buffers_)).data(),
                                    Spectral::Swsh::goldberg_mode_index(
                                        l_max_, libsharp_mode.l,
                                        -static_cast<int>(libsharp_mode.m))));
      });
    }
    // At this point, we have a collection of 9 tensors of libsharp
    // coefficients. This is what the boundary data calculation utility takes
    // as an input, so we now hand off the control flow to the boundary and
    // gauge transform utility
    if (buffer_updater_->radial_derivatives_need_renormalization()) {
      create_bondi_boundary_data_from_unnormalized_spec_modes(
          boundary_data_box,
          get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::SpatialMetric>>(
              interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
              interpolated_coefficients_),
          get<Tags::detail::Shift>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::Shift>>(
              interpolated_coefficients_),
          get<Tags::detail::Lapse>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::Lapse>>(
              interpolated_coefficients_),
          buffer_updater_->get_extraction_radius(), l_max_);
    } else {
      create_bondi_boundary_data(
          boundary_data_box,
          get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::SpatialMetric>>(
              interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
              interpolated_coefficients_),
          get<Tags::detail::Shift>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::Shift>>(
              interpolated_coefficients_),
          get<Tags::detail::Lapse>(interpolated_coefficients_),
          get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
          get<Tags::detail::Dr<Tags::detail::Lapse>>(
              interpolated_coefficients_),
          buffer_updater_->get_extraction_radius(), l_max_);
    }
    return true;
  }

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const noexcept { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const noexcept {
    return std::make_pair(time_span_start_, time_span_end_);
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept {  // NOLINT
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
      coefficients_buffers_ = Variables<detail::cce_input_tags>{size_of_buffer};
      interpolated_coefficients_ = Variables<detail::cce_input_tags>{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
    }
  }

 private:
  std::unique_ptr<WorldtubeBufferUpdater> buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<detail::cce_input_tags> interpolated_coefficients_;

  // note: buffers store data in a 'time-varies-fastest' manner
  mutable Variables<detail::cce_input_tags> coefficients_buffers_;

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
 * `detail::reduced_cce_input_tags`, rather than direct metric components
 * handled by `WorldtubeDataManager`. The set of 9 scalars is a far leaner
 * (factor of ~4) data storage format.
 */
class ReducedWorldtubeDataManager {
 public:
  // charm needs an empty constructor.
  ReducedWorldtubeDataManager() = default;

  ReducedWorldtubeDataManager(
      std::unique_ptr<ReducedWorldtubeBufferUpdater> buffer_updater,
      size_t l_max, size_t buffer_depth,
      std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept;

  /*!
   * \brief Update the `boundary_data_box` entries for all tags in
   * `Tags::characteristic_worldtube_boundary_tags` to the boundary data at
   * `time`.
   *
   * \details First, if the stored buffer requires updating, it will be updated
   * via the `buffer_updater_` supplied in the constructor. Then, each of the
   * 9 spin-weighted scalars in `detail::reduced_cce_input_tags`
   * are interpolated across buffer points to the requested time value (via the
   * `Interpolator` provided in the constructor). Finally, the remaining two
   * scalars not directly supplied in the input file are calculated in-line and
   * put in the \ref DataBoxGroup.
   *
   * Returns `true` if the time can be supplied from the `buffer_updater_`, and
   * `false` otherwise. No tags are updated if `false` is returned.
   */
  template <typename TagList>
  bool populate_hypersurface_boundary_data(
      gsl::not_null<db::DataBox<TagList>*> boundary_data_box, double time) const
      noexcept;

  /// retrieves the l_max that will be supplied to the \ref DataBoxGroup in
  /// `populate_hypersurface_boundary_data()`
  size_t get_l_max() const noexcept { return l_max_; }

  /// retrieves the current time span associated with the `buffer_updater_` for
  /// diagnostics
  std::pair<size_t, size_t> get_time_span() const noexcept {
    return std::make_pair(time_span_start_, time_span_end_);
  }

  /// Serialization for Charm++.
  void pup(PUP::er& p) noexcept;  // NOLINT

 private:
  std::unique_ptr<ReducedWorldtubeBufferUpdater> buffer_updater_;
  mutable size_t time_span_start_ = 0;
  mutable size_t time_span_end_ = 0;
  size_t l_max_ = 0;

  // These buffers are just kept around to avoid allocations; they're
  // updated every time a time is requested
  mutable Variables<detail::reduced_cce_input_tags> interpolated_coefficients_;

  // note: buffers store data in an 'time-varies-fastest' manner
  mutable Variables<detail::reduced_cce_input_tags> coefficients_buffers_;

  size_t buffer_depth_ = 0;

  std::unique_ptr<intrp::SpanInterpolator> interpolator_;
};

template <typename TagList>
bool ReducedWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<db::DataBox<TagList>*> boundary_data_box,
    const double time) const noexcept {
  if (buffer_updater_->time_is_outside_range(time)) {
    return false;
  }
  buffer_updater_->update_buffers_for_time(
      make_not_null(&coefficients_buffers_), make_not_null(&time_span_start_),
      make_not_null(&time_span_end_), time, l_max_,
      interpolator_->required_number_of_points_before_and_after(),
      buffer_depth_);
  auto interpolation_time_span = detail::create_span_for_time_value(
      time, 0, interpolator_->required_number_of_points_before_and_after(),
      time_span_start_, time_span_end_, buffer_updater_->get_time_buffer());

  // search through and find the two interpolation points the time point is
  // between. If we can, put the range for the interpolation centered on the
  // desired point. If that can't be done (near the start or the end of the
  // simulation), make the range terminated at the start or end of the cached
  // data and extending for the desired range in the other direction.
  const size_t buffer_span_size = time_span_end_ - time_span_start_;
  const size_t interpolation_span_size =
      interpolation_time_span.second - interpolation_time_span.first;

  DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column =
      [&time, &time_points, &buffer_span_size, &interpolation_time_span,
       &interpolation_span_size, this](auto data, size_t column) {
        const auto interp_val = interpolator_->interpolate(
            gsl::span<const double>(time_points.data(), time_points.size()),
            gsl::span<const std::complex<double>>(
                data + column * (buffer_span_size) +
                    (interpolation_time_span.first - time_span_start_),
                interpolation_span_size),
            time);
        return interp_val;
      };

  // the ComplexModalVectors should be provided from the buffer_updater_ in
  // 'Goldberg' format, so we iterate over modes and convert to libsharp
  // format.
  for (const auto& libsharp_mode :
       Spectral::Swsh::cached_coefficients_metadata(l_max_)) {
    tmpl::for_each<detail::reduced_cce_input_tags>([
      this, &libsharp_mode, &interpolate_from_column
    ](auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode,
          make_not_null(&get(get<tag>(interpolated_coefficients_))), 0,
          interpolate_from_column(
              get(get<tag>(coefficients_buffers_)).data().data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m))),
          interpolate_from_column(
              get(get<tag>(coefficients_buffers_)).data().data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max_, libsharp_mode.l,
                  -static_cast<int>(libsharp_mode.m))));
    });
  }
  // just inverse transform the 'direct' tags
  tmpl::for_each<tmpl::transform<detail::reduced_cce_input_tags,
                                 tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
      [this, &boundary_data_box](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        db::mutate<Tags::BoundaryValue<tag>>(
            boundary_data_box,
            [this](
                const gsl::not_null<
                    db::item_type<Cce::Tags::BoundaryValue<tag>>*>
                    boundary_value,
                const db::item_type<Spectral::Swsh::Tags::SwshTransform<tag>>&
                    modal_boundary_value) noexcept {
              Spectral::Swsh::inverse_swsh_transform(
                  l_max_, 1, make_not_null(&get(*boundary_value)),
                  get(modal_boundary_value));
            },
            get<Spectral::Swsh::Tags::SwshTransform<tag>>(
                interpolated_coefficients_));
      });

  db::mutate<Tags::BoundaryValue<Tags::DuRDividedByR>>(
      boundary_data_box,
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
             du_r_divided_by_r,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& du_r,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& r) noexcept {
        get(*du_r_divided_by_r) = get(du_r) / get(r);
      },
      db::get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_box));

  // there's only a couple of tags desired by the core computation that aren't
  // stored in the 'reduced' format, so we perform the remaining computation
  // in-line here.

  db::mutate<Tags::BoundaryValue<Tags::BondiH>>(
      boundary_data_box,
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 2>>*> h,
         const Scalar<SpinWeighted<ComplexDataVector, 2>>& du_j,
         const Scalar<SpinWeighted<ComplexDataVector, 2>>& dr_j,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>&
             du_r_divided_by_r) noexcept {
        get(*h) = get(du_j) + get(r) * get(du_r_divided_by_r) * get(dr_j);
      },
      db::get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*boundary_data_box));

  // \partial_r U:
  db::mutate<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
      boundary_data_box,
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> dr_u,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& beta,
         const Scalar<SpinWeighted<ComplexDataVector, 0>>& r,
         const Scalar<SpinWeighted<ComplexDataVector, 1>>& q,
         const Scalar<SpinWeighted<ComplexDataVector, 2>>& j) noexcept {
        // allocation
        SpinWeighted<ComplexDataVector, 0> k;
        k.data() = sqrt(1.0 + get(j).data() * conj(get(j).data()));
        get(*dr_u).data() =
            exp(2.0 * get(beta).data()) / square(get(r).data()) *
            (k.data() * get(q).data() - get(j).data() * conj(get(q).data()));
      },
      db::get<Tags::BoundaryValue<Tags::BondiBeta>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::BondiQ>>(*boundary_data_box),
      db::get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_box));
  return true;
}
}  // namespace Cce
