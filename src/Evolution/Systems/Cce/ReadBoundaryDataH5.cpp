// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/ReadBoundaryDataH5.hpp"

#include <algorithm>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Assert.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Numeric.hpp"

namespace Cce {

namespace detail {
std::pair<size_t, size_t> create_span_for_time_value(
    const double time, const size_t pad, const size_t interpolator_length,
    const size_t lower_bound, const size_t upper_bound,
    const DataVector& time_buffer) noexcept {
  ASSERT(
      lower_bound < upper_bound,
      "The supplied `lower_bound` is greater than `upper_bound`, which is not "
      "permitted");
  ASSERT(2 * interpolator_length + pad < upper_bound,
         "The combined `interpolator_length` and `pad` is too large for the "
         "supplied `upper_bound`");

  size_t range_start = lower_bound;
  size_t range_end = upper_bound;
  while (range_end - range_start > 1) {
    if (time_buffer[(range_start + range_end) / 2] < time) {
      range_start = (range_start + range_end) / 2;
    } else {
      range_end = (range_start + range_end) / 2;
    }
  }
  // always keep the difference between start and end the same, even when
  // the interpolations starts to get worse
  size_t span_start = lower_bound;
  size_t span_end =
      std::min(interpolator_length * 2 + pad + lower_bound, upper_bound);
  if (range_end + interpolator_length + pad > upper_bound) {
    span_start =
        std::max(upper_bound - (interpolator_length * 2 + pad), lower_bound);
    span_end = upper_bound;
  } else if (range_start + 1 > lower_bound + interpolator_length) {
    span_start = range_start - interpolator_length;
    span_end = range_end + interpolator_length + pad - 1;
  }

  return std::make_pair(span_start, span_end);
}
}  // namespace detail

MetricWorldtubeH5BufferUpdater::MetricWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename) noexcept
    : cce_data_file_{cce_data_filename}, filename_{cce_data_filename} {
  get<Tags::detail::InputDataSet<Tags::detail::SpatialMetric>>(dataset_names_) =
      "/g";
  get<Tags::detail::InputDataSet<
      Tags::detail::Dr<Tags::detail::SpatialMetric>>>(dataset_names_) = "/Drg";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::SpatialMetric>>>(
      dataset_names_) = "/Dtg";

  get<Tags::detail::InputDataSet<Tags::detail::Shift>>(dataset_names_) =
      "/Shift";
  get<Tags::detail::InputDataSet<Tags::detail::Dr<Tags::detail::Shift>>>(
      dataset_names_) = "/DrShift";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::Shift>>>(
      dataset_names_) = "/DtShift";

  get<Tags::detail::InputDataSet<Tags::detail::Lapse>>(dataset_names_) =
      "/Lapse";
  get<Tags::detail::InputDataSet<Tags::detail::Dr<Tags::detail::Lapse>>>(
      dataset_names_) = "/DrLapse";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::Lapse>>>(
      dataset_names_) = "/DtLapse";

  // 'VersionHist' is a feature written by SpEC to indicate the details of the
  // file format. This line determines whether or not the radial derivatives
  // require renormalization based on whether the SpEC version that produced it
  // was an old one that had a particular normalization bug
  radial_derivatives_need_renormalization_ =
      not cce_data_file_.exists<h5::Version>("/VersionHist");

  // We assume that the filename has the extraction radius encoded as an
  // integer between the last occurrence of 'R' and the last occurrence of
  // '.'. This is the format provided by SpEC.
  const size_t r_pos = cce_data_filename.find_last_of('R');
  const size_t dot_pos = cce_data_filename.find_last_of('.');
  const std::string text_radius =
      cce_data_filename.substr(r_pos + 1, dot_pos - r_pos - 1);
  try {
    extraction_radius_ = stod(text_radius);
  } catch (const std::invalid_argument&) {
    ERROR(
        "The CCE filename must encode the extraction radius as an integer "
        "between the first instance of 'R' and the first instance of '.' (SpEC "
        "CCE filename format). Provided filename : "
        << cce_data_filename);
  }
  const auto& lapse_data = cce_data_file_.get<h5::Dat>("/Lapse");
  const auto data_table_dimensions = lapse_data.get_dimensions();
  const Matrix time_matrix = lapse_data.get_data_subset(
      std::vector<size_t>{0}, 0, data_table_dimensions[0]);
  time_buffer_ = DataVector{data_table_dimensions[0]};
  for (size_t i = 0; i < data_table_dimensions[0]; ++i) {
    time_buffer_[i] = time_matrix(i, 0);
  }
  l_max_ = sqrt(data_table_dimensions[1] / 2) - 1;
  cce_data_file_.close_current_object();
}

double MetricWorldtubeH5BufferUpdater::update_buffers_for_time(
    const gsl::not_null<Variables<cce_input_tags>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth) const noexcept {
  if (*time_span_end >= time_buffer_.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (*time_span_end > interpolator_length and
      time_buffer_[*time_span_end - interpolator_length] > time) {
    // the next time an update will be required
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }
  // find the time spans that are needed
  auto new_span_pair = detail::create_span_for_time_value(
      time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
      time_buffer_);
  *time_span_start = new_span_pair.first;
  *time_span_end = new_span_pair.second;
  // load the desired time spans into the buffers
  // spatial metric
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      tmpl::for_each<tmpl::list<Tags::detail::SpatialMetric,
                                Tags::detail::Dr<Tags::detail::SpatialMetric>,
                                ::Tags::dt<Tags::detail::SpatialMetric>>>(
          [this, &i, &j, &buffers, &time_span_start, &time_span_end,
           &computation_l_max](auto tag_v) noexcept {
            using tag = typename decltype(tag_v)::type;
            this->update_buffer(
                make_not_null(&get<tag>(*buffers).get(i, j)),
                cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                    get<Tags::detail::InputDataSet<tag>>(dataset_names_), i,
                    j)),
                computation_l_max, *time_span_start, *time_span_end);
            cce_data_file_.close_current_object();
          });
    }
    // shift
    tmpl::for_each<
        tmpl::list<Tags::detail::Shift, Tags::detail::Dr<Tags::detail::Shift>,
                   ::Tags::dt<Tags::detail::Shift>>>(
        [this, &i, &buffers, &time_span_start, &time_span_end,
         &computation_l_max](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          this->update_buffer(
              make_not_null(&get<tag>(*buffers).get(i)),
              cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                  get<Tags::detail::InputDataSet<tag>>(dataset_names_), i)),
              computation_l_max, *time_span_start, *time_span_end);
          cce_data_file_.close_current_object();
        });
  }
  // lapse
  tmpl::for_each<
      tmpl::list<Tags::detail::Lapse, Tags::detail::Dr<Tags::detail::Lapse>,
                 ::Tags::dt<Tags::detail::Lapse>>>(
      [this, &buffers, &time_span_start, &time_span_end,
       &computation_l_max](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        this->update_buffer(
            make_not_null(&get(get<tag>(*buffers))),
            cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                get<Tags::detail::InputDataSet<tag>>(dataset_names_))),
            computation_l_max, *time_span_start, *time_span_end);
        cce_data_file_.close_current_object();
      });
  // the next time an update will be required
  return time_buffer_[std::min(*time_span_end - interpolator_length + 1,
                               time_buffer_.size() - 1)];
}

std::unique_ptr<WorldtubeBufferUpdater<cce_input_tags>>
MetricWorldtubeH5BufferUpdater::get_clone() const noexcept {
  return std::make_unique<MetricWorldtubeH5BufferUpdater>(
      MetricWorldtubeH5BufferUpdater{filename_});
}

bool MetricWorldtubeH5BufferUpdater::time_is_outside_range(
    const double time) const noexcept {
  return time < time_buffer_[0] or time > time_buffer_[time_buffer_.size() - 1];
}

void MetricWorldtubeH5BufferUpdater::pup(PUP::er& p) noexcept {
  p | time_buffer_;
  p | radial_derivatives_need_renormalization_;
  p | filename_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

void MetricWorldtubeH5BufferUpdater::update_buffer(
    const gsl::not_null<ComplexModalVector*> buffer_to_update,
    const h5::Dat& read_data, const size_t computation_l_max,
    const size_t time_span_start, const size_t time_span_end) const noexcept {
  const size_t number_of_columns = read_data.get_dimensions()[1];
  if (UNLIKELY(buffer_to_update->size() != (time_span_end - time_span_start) *
                                               square(computation_l_max + 1))) {
    ERROR("Incorrect storage size for the data to be loaded in.");
  }
  auto cols = alg::iota(std::vector<size_t>(number_of_columns - 1), 1_st);
  const Matrix data_matrix = read_data.get_data_subset(
      cols, time_span_start, time_span_end - time_span_start);

  *buffer_to_update = 0.0;
  for (size_t time_row = 0; time_row < time_span_end - time_span_start;
       ++time_row) {
    for (int l = 0; l <= static_cast<int>(std::min(computation_l_max, l_max_));
         ++l) {
      for (int m = -l; m <= l; ++m) {
        (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                computation_l_max, static_cast<size_t>(l), m) *
                                (time_span_end - time_span_start) +
                            time_row] =
            // -m because SpEC format is stored in decending m.
            std::complex<double>(
                data_matrix(time_row,
                            2 * Spectral::Swsh::goldberg_mode_index(
                                    l_max_, static_cast<size_t>(l), -m)),
                data_matrix(time_row,
                            2 * Spectral::Swsh::goldberg_mode_index(
                                    l_max_, static_cast<size_t>(l), -m) +
                                1));
      }
    }
  }
}

BondiWorldtubeH5BufferUpdater::BondiWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename) noexcept
    : cce_data_file_{cce_data_filename}, filename_{cce_data_filename} {
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiBeta>>>(dataset_names_) =
      "Beta";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiU>>>(dataset_names_) = "U";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiQ>>>(dataset_names_) = "Q";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiW>>>(dataset_names_) = "W";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiJ>>>(dataset_names_) = "J";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::Dr<Tags::BondiJ>>>>(
      dataset_names_) = "DrJ";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::Du<Tags::BondiJ>>>>(
      dataset_names_) = "H";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::BondiR>>>(dataset_names_) = "R";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::Du<Tags::BondiR>>>>(
      dataset_names_) = "DuR";

  // We assume that the filename has the extraction radius encoded as an
  // integer between the last occurrence of 'R' and the last occurrence of
  // '.'. This is the format provided by SpEC.
  const size_t r_pos = cce_data_filename.find_last_of('R');
  const size_t dot_pos = cce_data_filename.find_last_of('.');
  const std::string text_radius =
      cce_data_filename.substr(r_pos + 1, dot_pos - r_pos - 1);
  try {
    extraction_radius_ = stod(text_radius);
  } catch (const std::invalid_argument&) {
    ERROR(
        "The CCE filename must encode the extraction radius as an integer "
        "between the first instance of 'R' and the first instance of '.' (SpEC "
        "CCE filename format). Provided filename : "
        << cce_data_filename);
  }

  const auto& u_data = cce_data_file_.get<h5::Dat>("/U");
  const auto data_table_dimensions = u_data.get_dimensions();
  const Matrix time_matrix = u_data.get_data_subset(std::vector<size_t>{0}, 0,
                                                    data_table_dimensions[0]);
  time_buffer_ = DataVector{data_table_dimensions[0]};
  for (size_t i = 0; i < data_table_dimensions[0]; ++i) {
    time_buffer_[i] = time_matrix(i, 0);
  }
  l_max_ = sqrt(data_table_dimensions[1] / 2) - 1;
  cce_data_file_.close_current_object();
}

double BondiWorldtubeH5BufferUpdater::update_buffers_for_time(
    const gsl::not_null<Variables<reduced_cce_input_tags>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth) const noexcept {
  if (*time_span_end >= time_buffer_.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (*time_span_end > interpolator_length and
      time_buffer_[*time_span_end - interpolator_length] > time) {
    // the next time an update will be required
    return time_buffer_[*time_span_end - interpolator_length + 1];
  }
  // find the time spans that are needed
  auto new_span_pair = detail::create_span_for_time_value(
      time, buffer_depth, interpolator_length, 0, time_buffer_.size(),
      time_buffer_);
  *time_span_start = new_span_pair.first;
  *time_span_end = new_span_pair.second;
  // load the desired time spans into the buffers
  tmpl::for_each<reduced_cce_input_tags>(
      [this, &buffers, &time_span_start, &time_span_end,
       &computation_l_max](auto tag_v) noexcept {
        using tag = typename decltype(tag_v)::type;
        this->update_buffer(
            make_not_null(&get(get<tag>(*buffers)).data()),
            cce_data_file_.get<h5::Dat>(
                "/" + get<Tags::detail::InputDataSet<tag>>(dataset_names_)),
            computation_l_max, *time_span_start, *time_span_end,
            tag::type::type::spin == 0);
        cce_data_file_.close_current_object();
      });
  // the next time an update will be required
  return time_buffer_[std::min(*time_span_end - interpolator_length + 1,
                               time_buffer_.size() - 1)];
}

void BondiWorldtubeH5BufferUpdater::update_buffer(
    const gsl::not_null<ComplexModalVector*> buffer_to_update,
    const h5::Dat& read_data, const size_t computation_l_max,
    const size_t time_span_start, const size_t time_span_end,
    const bool is_real) const noexcept {
  size_t number_of_columns = read_data.get_dimensions()[1];
  if (UNLIKELY(buffer_to_update->size() !=
               square(computation_l_max + 1) *
                   (time_span_end - time_span_start))) {
    ERROR("Incorrect storage size for the data to be loaded in.");
  }
  std::vector<size_t> cols(number_of_columns - 1);
  std::iota(cols.begin(), cols.end(), 1);
  Matrix data_matrix = read_data.get_data_subset(
      cols, time_span_start, time_span_end - time_span_start);
  *buffer_to_update = 0.0;
  for (size_t time_row = 0; time_row < time_span_end - time_span_start;
       ++time_row) {
    for (int l = 0; l <= static_cast<int>(std::min(computation_l_max, l_max_));
         ++l) {
      for (int m = -l; m <= l; ++m) {
        if (is_real) {
          if (m == 0) {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                std::complex<double>(
                    data_matrix(time_row, static_cast<size_t>(square(l))), 0.0);
          } else if (m > 0) {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                std::complex<double>(
                    data_matrix(time_row,
                                static_cast<size_t>(square(l) + 2 * m - 1)),
                    data_matrix(
                        time_row,
                        static_cast<size_t>(square(l) + 2 * m)));  // NOLINT
          } else {
            (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                    computation_l_max, static_cast<size_t>(l),
                                    m) *
                                    (time_span_end - time_span_start) +
                                time_row] =
                (-m % 2 == 0 ? 1.0 : -1.0) *
                std::complex<double>(
                    data_matrix(time_row,
                                static_cast<size_t>(square(l) + 2 * -m - 1)),
                    -data_matrix(
                        time_row,
                        static_cast<size_t>(square(l) + 2 * -m)));  // NOLINT
          }
        } else {
          (*buffer_to_update)[Spectral::Swsh::goldberg_mode_index(
                                  computation_l_max, static_cast<size_t>(l),
                                  m) *
                                  (time_span_end - time_span_start) +
                              time_row] =
              std::complex<double>(
                  data_matrix(time_row,
                              2 * Spectral::Swsh::goldberg_mode_index(
                                      l_max_, static_cast<size_t>(l), m)),
                  data_matrix(time_row,
                              2 * Spectral::Swsh::goldberg_mode_index(
                                      l_max_, static_cast<size_t>(l), m) +
                                  1));
        }
      }
    }
  }
}

void BondiWorldtubeH5BufferUpdater::pup(PUP::er& p) noexcept {
  p | time_buffer_;
  p | filename_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

bool MetricWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
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

  const DataVector time_points{
      buffer_updater_->get_time_buffer().data() + interpolation_time_span.first,
      interpolation_span_size};

  auto interpolate_from_column = [&time, &time_points, &buffer_span_size,
                                  &interpolation_time_span,
                                  &interpolation_span_size, this](
                                     auto data, const size_t column) noexcept {
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
        tmpl::for_each<tmpl::list<Tags::detail::SpatialMetric,
                                  Tags::detail::Dr<Tags::detail::SpatialMetric>,
                                  ::Tags::dt<Tags::detail::SpatialMetric>>>(
            [this, &i, &j, &libsharp_mode, &interpolate_from_column,
             &spin_weighted_buffer](auto tag_v) noexcept {
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
      tmpl::for_each<
          tmpl::list<Tags::detail::Shift, Tags::detail::Dr<Tags::detail::Shift>,
                     ::Tags::dt<Tags::detail::Shift>>>(
          [this, &i, &libsharp_mode, &interpolate_from_column,
           &spin_weighted_buffer](auto tag_v) noexcept {
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
                   ::Tags::dt<Tags::detail::Lapse>>>([this, &libsharp_mode,
                                                      &interpolate_from_column,
                                                      &spin_weighted_buffer](
                                                         auto tag_v) noexcept {
      using tag = typename decltype(tag_v)::type;
      spin_weighted_buffer.set_data_ref(
          get(get<tag>(interpolated_coefficients_)).data(),
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_));
      Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
          libsharp_mode, make_not_null(&spin_weighted_buffer), 0,
          interpolate_from_column(
              get(get<tag>(coefficients_buffers_)).data(),
              Spectral::Swsh::goldberg_mode_index(
                  l_max_, libsharp_mode.l, static_cast<int>(libsharp_mode.m))),
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
        boundary_data_variables,
        get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Shift>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Lapse>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Lapse>>(interpolated_coefficients_),
        buffer_updater_->get_extraction_radius(), l_max_);
  } else {
    create_bondi_boundary_data(
        boundary_data_variables,
        get<Tags::detail::SpatialMetric>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::SpatialMetric>>(
            interpolated_coefficients_),
        get<Tags::detail::Shift>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Shift>>(interpolated_coefficients_),
        get<Tags::detail::Lapse>(interpolated_coefficients_),
        get<::Tags::dt<Tags::detail::Lapse>>(interpolated_coefficients_),
        get<Tags::detail::Dr<Tags::detail::Lapse>>(interpolated_coefficients_),
        buffer_updater_->get_extraction_radius(), l_max_);
  }
  return true;
}

BondiWorldtubeDataManager::BondiWorldtubeDataManager(
    std::unique_ptr<WorldtubeBufferUpdater<reduced_cce_input_tags>>
        buffer_updater,
    const size_t l_max, const size_t buffer_depth,
    std::unique_ptr<intrp::SpanInterpolator> interpolator) noexcept
    : buffer_updater_{std::move(buffer_updater)},
      l_max_{l_max},
      interpolated_coefficients_{
          Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max)},
      buffer_depth_{buffer_depth},
      interpolator_{std::move(interpolator)} {
  if (UNLIKELY(buffer_updater_->get_time_buffer().size() <
               2 * interpolator_->required_number_of_points_before_and_after() +
                   buffer_depth)) {
    ERROR(
        "The specified buffer updater doesn't have enough time points to "
        "supply the requested interpolation buffer. This almost certainly "
        "indicates that the corresponding file hasn't been created properly, "
        "but might indicate that the `buffer_depth` template parameter is "
        "too large or the specified SpanInterpolator requests too many "
        "points");
  }
  coefficients_buffers_ = Variables<reduced_cce_input_tags>{
      square(l_max + 1) *
      (buffer_depth +
       2 * interpolator_->required_number_of_points_before_and_after())};
}

bool BondiWorldtubeDataManager::populate_hypersurface_boundary_data(
    const gsl::not_null<Variables<
        Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>>*>
        boundary_data_variables,
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
    tmpl::for_each<reduced_cce_input_tags>(
        [this, &libsharp_mode, &interpolate_from_column](auto tag_v) noexcept {
          using tag = typename decltype(tag_v)::type;
          Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
              libsharp_mode,
              make_not_null(&get(get<tag>(interpolated_coefficients_))), 0,
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      static_cast<int>(libsharp_mode.m))),
              interpolate_from_column(
                  get(get<tag>(coefficients_buffers_)).data().data(),
                  Spectral::Swsh::goldberg_mode_index(
                      l_max_, libsharp_mode.l,
                      -static_cast<int>(libsharp_mode.m))));
        });
  }
  // just inverse transform the 'direct' tags
  tmpl::for_each<tmpl::transform<reduced_cce_input_tags,
                                 tmpl::bind<db::remove_tag_prefix, tmpl::_1>>>(
      [this, &boundary_data_variables](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        Spectral::Swsh::inverse_swsh_transform(
            l_max_, 1,
            make_not_null(
                &get(get<Tags::BoundaryValue<tag>>(*boundary_data_variables))),
            get(get<Spectral::Swsh::Tags::SwshTransform<tag>>(
                interpolated_coefficients_)));
      });
  const auto& du_r = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiR>>>(
      *boundary_data_variables));
  const auto& bondi_r =
      get(get<Tags::BoundaryValue<Tags::BondiR>>(*boundary_data_variables));

  get(get<Tags::BoundaryValue<Tags::DuRDividedByR>>(*boundary_data_variables)) =
      du_r / bondi_r;

  // there's only a couple of tags desired by the core computation that aren't
  // stored in the 'reduced' format, so we perform the remaining computation
  // in-line here.
  const auto& du_bondi_j = get(get<Tags::BoundaryValue<Tags::Du<Tags::BondiJ>>>(
      *boundary_data_variables));
  const auto& dr_bondi_j = get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiJ>>>(
      *boundary_data_variables));
  get(get<Tags::BoundaryValue<Tags::BondiH>>(*boundary_data_variables)) =
      du_bondi_j + du_r * dr_bondi_j;

  const auto& bondi_j =
      get(get<Tags::BoundaryValue<Tags::BondiJ>>(*boundary_data_variables));
  const auto& bondi_beta =
      get(get<Tags::BoundaryValue<Tags::BondiBeta>>(*boundary_data_variables));
  const auto& bondi_q =
      get(get<Tags::BoundaryValue<Tags::BondiQ>>(*boundary_data_variables));
  const auto& bondi_k = sqrt(1.0 + bondi_j * conj(bondi_j));
  get(get<Tags::BoundaryValue<Tags::Dr<Tags::BondiU>>>(
      *boundary_data_variables)) =
      exp(2.0 * bondi_beta.data()) / square(bondi_r.data()) *
      (bondi_k.data() * bondi_q.data() - bondi_j.data() * conj(bondi_q.data()));
  return true;
}

void BondiWorldtubeDataManager::pup(PUP::er& p) noexcept {
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
    coefficients_buffers_ = Variables<reduced_cce_input_tags>{size_of_buffer};
    interpolated_coefficients_ = Variables<reduced_cce_input_tags>{
        Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  }
}

/// \cond
PUP::able::PUP_ID MetricWorldtubeH5BufferUpdater::my_PUP_ID = 0;
PUP::able::PUP_ID BondiWorldtubeH5BufferUpdater::my_PUP_ID = 0;
PUP::able::PUP_ID MetricWorldtubeDataManager::my_PUP_ID = 0;
PUP::able::PUP_ID BondiWorldtubeDataManager::my_PUP_ID = 0;
/// \endcond
}  // namespace Cce
