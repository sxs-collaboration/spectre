// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeBufferUpdater.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/Cce/ExtractionRadius.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Version.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Math.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace detail {
std::pair<size_t, size_t> create_span_for_time_value(
    const double time, const size_t pad, const size_t interpolator_length,
    const size_t lower_bound, const size_t upper_bound,
    const DataVector& time_buffer) {
  ASSERT(
      lower_bound < upper_bound,
      "The supplied `lower_bound` is greater than `upper_bound`, which is not "
      "permitted");
  ASSERT(2 * interpolator_length + pad <= upper_bound,
         "The combined `interpolator_length` and `pad` is too large for the "
         "supplied `upper_bound`.\nupper_bound="
             << upper_bound << "\npad=" << pad
             << "\ninterpolator_length=" << interpolator_length);

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

void set_time_buffer_and_lmax(const gsl::not_null<DataVector*> time_buffer,
                              size_t& l_max, const h5::Dat& data,
                              const bool is_real, const bool is_modal_data,
                              const bool is_complex) {
  const auto data_table_dimensions = data.get_dimensions();
  const Matrix time_matrix =
      data.get_data_subset(std::vector<size_t>{0}, 0, data_table_dimensions[0]);
  *time_buffer = DataVector{data_table_dimensions[0]};

  for (size_t i = 0; i < data_table_dimensions[0]; ++i) {
    (*time_buffer)[i] = time_matrix(i, 0);
  }

  if (is_modal_data) {
    // If the quantitiy is real, then there's no way for us to do a check just
    // from the number of columns alone. If not, then we expect an even number
    // of total modes (both real and imaginary), so the number of columns should
    // be odd with the addition of the time column
    if (not is_real and data_table_dimensions[1] % 2 != 1) {
      ERROR("Dimensions of subfile "
            << data.subfile_path()
            << " for modal data are incorrect. Was expecting an odd number of "
               "columns (because of the time), but got "
            << data_table_dimensions[1] << " instead.");
    }

    // If it's real the number of columns is (l+1)^2. If it's not, then it's
    // 2(l+1)^2
    const size_t l_plus_one_squared =
        (data_table_dimensions[1] - 1) / (is_real ? 1 : 2);
    l_max =
        static_cast<size_t>(sqrt(static_cast<double>(l_plus_one_squared)) - 1);
  } else {
    // Can only check number of columns for nodal data if it's complex
    if (is_complex and data_table_dimensions[1] % 2 != 1) {
      ERROR("Dimensions of subfile "
            << data.subfile_path()
            << " for nodal data are incorrect. Was expecting an odd number of "
               "columns (because of the time), but got "
            << data_table_dimensions[1] << " instead.");
    }

    // If number of real values N = (l+1)*(2l+1), then
    // l = ( -3 + sqrt(9 + 8 * (N-1)) ) / 4. But the dimensions[1] includes time
    // so we have to account for that by subtracting 1. Also, if this is complex
    // nodal data, we have to do N=(dimensions[1]-1)/2 first
    const size_t num_real_values = is_complex
                                       ? (data_table_dimensions[1] - 1) / 2
                                       : data_table_dimensions[1] - 1;
    if (num_real_values == 0) {
      ERROR("Not enough columns to read "
            << (is_complex ? "complex" : "real")
            << " nodal data from. Number of columns in file ("
            << data_table_dimensions[1]
            << "), number of columns without time column ("
            << data_table_dimensions[1] << "), number of real valued columns ("
            << num_real_values << ")");
    }
    const size_t discriminant = 9 + 8 * (num_real_values - 1);
    l_max =
        static_cast<size_t>((sqrt(static_cast<double>(discriminant)) - 3)) / 4;
  }
}

template <bool IsModal, int Spin, typename T>
void update_buffer(const gsl::not_null<T*> buffer_to_update,
                   const h5::Dat& read_data, const size_t computation_l_max,
                   const size_t l_max, const size_t time_span_start,
                   const size_t time_span_end,
                   const bool time_varies_fastest = true) {
  constexpr bool is_real = Spin == 0;
  size_t number_of_columns = read_data.get_dimensions()[1];
  const size_t result_l_max = std::min(l_max, computation_l_max);
  // No x2 because the datatypes are std::complex
  const size_t expected_size =
      (time_span_end - time_span_start) *
      (IsModal ? square(computation_l_max + 1)
               : Spectral::Swsh::number_of_swsh_collocation_points(
                     computation_l_max));

  if (UNLIKELY(buffer_to_update->size() != expected_size)) {
    ERROR("Incorrect storage size for the data to be loaded in. Expected "
          << expected_size << ", but got " << buffer_to_update->size()
          << " instead.");
  }

  std::vector<size_t> cols(number_of_columns - 1);
  std::iota(cols.begin(), cols.end(), 1);
  auto data_matrix =
      read_data.get_data_subset<std::vector<std::vector<double>>>(
          cols, time_span_start, time_span_end - time_span_start);
  *buffer_to_update = 0.0;

  // For modal, only x2 if it's not real. If nodal, always x2
  const size_t expected_dat_modal_size = (is_real ? 1 : 2) * square(l_max + 1);
  const size_t expected_dat_nodal_size =
      2 * Spectral::Swsh::number_of_swsh_collocation_points(l_max);

  if (cols.size() !=
      (IsModal ? expected_dat_modal_size : expected_dat_nodal_size)) {
    ERROR("Incorrect number of columns in Dat file "
          << read_data.subfile_path() << ". Expected (excluding time column) "
          << (IsModal ? expected_dat_modal_size : expected_dat_nodal_size)
          << ", but got " << cols.size());
  }

  for (size_t time_row = 0; time_row < time_span_end - time_span_start;
       ++time_row) {
    // For modal data we have to compute the correct goldberg mode indices for
    // each l and m
    if constexpr (IsModal) {
      for (int l = 0; l <= static_cast<int>(result_l_max); ++l) {
        for (int m = -l; m <= l; ++m) {
          const size_t buffer_index =
              time_varies_fastest
                  ? Spectral::Swsh::goldberg_mode_index(
                        computation_l_max, static_cast<size_t>(l), m) *
                            (time_span_end - time_span_start) +
                        time_row
                  : time_row * square(computation_l_max + 1) +
                        Spectral::Swsh::goldberg_mode_index(
                            computation_l_max, static_cast<size_t>(l), m);
          // NOLINTBEGIN
          // If this quantity is real we don't expect to store the imaginary m=0
          // modes in the H5 file so those have to be handled specially. Then
          // for +/- and even/odd m-modes we have to have the correct sign for
          // the coefficient.
          if (is_real) {
            if (m == 0) {
              (*buffer_to_update)[buffer_index] = std::complex<double>(
                  data_matrix[time_row][static_cast<size_t>(square(l))], 0.0);
            } else {
              const double factor = (m > 0 or abs(m) % 2 == 0) ? 1.0 : -1.0;
              const size_t matrix_index =
                  static_cast<size_t>(square(l) + 2 * abs(m));
              (*buffer_to_update)[buffer_index] =
                  factor * std::complex<double>(
                               data_matrix[time_row][matrix_index - 1],
                               sgn(m) * data_matrix[time_row][matrix_index]);
            }
          } else {
            // If this quantity is complex, then it's straight forward to
            // populate the buffer
            const size_t matrix_goldberg_index =
                Spectral::Swsh::goldberg_mode_index(l_max,
                                                    static_cast<size_t>(l), m);
            (*buffer_to_update)[buffer_index] = std::complex<double>(
                data_matrix[time_row][2 * matrix_goldberg_index],
                data_matrix[time_row][2 * matrix_goldberg_index + 1]);
          }
          // NOLINTEND
        }
      }
    } else {
      (void)result_l_max;
      // For nodal data, it must be complex so for each grid point we just read
      // in the real and imaginary components
      const size_t number_of_angular_points =
          Spectral::Swsh::number_of_swsh_collocation_points(l_max);
      for (size_t i = 0; i < number_of_angular_points; i++) {
        const size_t buffer_index =
            time_varies_fastest
                ? i * (time_span_end - time_span_start) + time_row
                : time_row * number_of_angular_points + i;
        (*buffer_to_update)[buffer_index] = std::complex<double>(
            data_matrix[time_row][2 * i], data_matrix[time_row][2 * i + 1]);
      }
    }
  }
}

template <bool IsModal, typename InputTags>
double update_buffers_for_time(
    const gsl::not_null<Variables<InputTags>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t l_max,
    const size_t interpolator_length, const size_t buffer_depth,
    const DataVector& time_buffer,
    const tuples::tagged_tuple_from_typelist<
        db::wrap_tags_in<Tags::detail::InputDataSet, InputTags>>& dataset_names,
    const h5::H5File<h5::AccessType::ReadOnly>& cce_data_file,
    const bool time_varies_fastest = true) {
  if (not IsModal and computation_l_max != l_max) {
    ERROR(
        "When reading in nodal data, the LMax that "
        "the BufferUpdater was constructed with ("
        << l_max
        << ") must be the same as the computation LMax passed to the "
           "update_buffers_for_time function ("
        << computation_l_max << ").");
  }
  if (*time_span_end >= time_buffer.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (*time_span_end > interpolator_length and
      time_buffer[*time_span_end - interpolator_length] > time) {
    // the next time an update will be required
    return time_buffer[*time_span_end - interpolator_length + 1];
  }
  // find the time spans that are needed
  auto new_span_pair = detail::create_span_for_time_value(
      time, buffer_depth, interpolator_length, 0, time_buffer.size(),
      time_buffer);
  *time_span_start = new_span_pair.first;
  *time_span_end = new_span_pair.second;
  // load the desired time spans into the buffers
  tmpl::for_each<InputTags>([&](auto tag_v) {
    using tag = typename decltype(tag_v)::type;
    update_buffer<IsModal, tag::type::type::spin>(
        make_not_null(&get(get<tag>(*buffers)).data()),
        cce_data_file.get<h5::Dat>(
            "/" + get<Tags::detail::InputDataSet<tag>>(dataset_names)),
        computation_l_max, l_max, *time_span_start, *time_span_end,
        time_varies_fastest);
    cce_data_file.close_current_object();
  });
  // the next time an update will be required
  return time_buffer[std::min(*time_span_end - interpolator_length + 1,
                              time_buffer.size() - 1)];
}

}  // namespace detail

template <typename T>
MetricWorldtubeH5BufferUpdater<T>::MetricWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename,
    const std::optional<double> extraction_radius, const bool descending_m)
    : cce_data_file_{cce_data_filename},
      filename_{cce_data_filename},
      descending_m_(descending_m) {
  get<Tags::detail::InputDataSet<Tags::detail::SpatialMetric<T>>>(
      dataset_names_) = "/g";
  get<Tags::detail::InputDataSet<
      Tags::detail::Dr<Tags::detail::SpatialMetric<T>>>>(dataset_names_) =
      "/Drg";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::SpatialMetric<T>>>>(
      dataset_names_) = "/Dtg";

  get<Tags::detail::InputDataSet<Tags::detail::Shift<T>>>(dataset_names_) =
      "/Shift";
  get<Tags::detail::InputDataSet<Tags::detail::Dr<Tags::detail::Shift<T>>>>(
      dataset_names_) = "/DrShift";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::Shift<T>>>>(
      dataset_names_) = "/DtShift";

  get<Tags::detail::InputDataSet<Tags::detail::Lapse<T>>>(dataset_names_) =
      "/Lapse";
  get<Tags::detail::InputDataSet<Tags::detail::Dr<Tags::detail::Lapse<T>>>>(
      dataset_names_) = "/DrLapse";
  get<Tags::detail::InputDataSet<::Tags::dt<Tags::detail::Lapse<T>>>>(
      dataset_names_) = "/DtLapse";

  // 'VersionHist' is a feature written by SpEC to indicate the details of the
  // file format. This line determines whether or not the radial derivatives
  // require renormalization based on whether the SpEC version that produced it
  // was an old one that had a particular normalization bug
  has_version_history_ = cce_data_file_.exists<h5::Version>("/VersionHist");

  extraction_radius_ =
      Cce::get_extraction_radius(cce_data_filename, extraction_radius, true)
          .value();

  detail::set_time_buffer_and_lmax(make_not_null(&time_buffer_), l_max_,
                                   cce_data_file_.get<h5::Dat>("/Lapse"), false,
                                   is_modal, is_modal);
  cce_data_file_.close_current_object();
}

template <typename T>
double MetricWorldtubeH5BufferUpdater<T>::update_buffers_for_time(
    const gsl::not_null<Variables<cce_metric_input_tags<T>>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth, const bool time_varies_fastest) const {
  // We require these to be the same for nodal data since we can't easily
  // extend nodal data like we can modal data.
  if (not is_modal and computation_l_max != l_max_) {
    ERROR(
        "When reading in nodal data, the LMax that "
        "MetricWorldtubeH5BufferUpdater was constructed with ("
        << l_max_
        << ") must be the same as the computation LMax passed to the "
           "update_buffers_for_time function ("
        << computation_l_max << ").");
  }
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
      tmpl::for_each<
          Tags::detail::apply_derivs_t<Tags::detail::SpatialMetric<T>>>(
          [&, this](auto tag_v) {
            using tag = typename decltype(tag_v)::type;
            this->update_buffer(
                make_not_null(&get<tag>(*buffers).get(i, j)),
                cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                    get<Tags::detail::InputDataSet<tag>>(dataset_names_), i,
                    j)),
                computation_l_max, *time_span_start, *time_span_end,
                time_varies_fastest);
            cce_data_file_.close_current_object();
          });
    }
    // shift
    tmpl::for_each<Tags::detail::apply_derivs_t<Tags::detail::Shift<T>>>(
        [&, this](auto tag_v) {
          using tag = typename decltype(tag_v)::type;
          this->update_buffer(
              make_not_null(&get<tag>(*buffers).get(i)),
              cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                  get<Tags::detail::InputDataSet<tag>>(dataset_names_), i)),
              computation_l_max, *time_span_start, *time_span_end,
              time_varies_fastest);
          cce_data_file_.close_current_object();
        });
  }
  // lapse
  tmpl::for_each<Tags::detail::apply_derivs_t<Tags::detail::Lapse<T>>>(
      [&, this](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        this->update_buffer(
            make_not_null(&get(get<tag>(*buffers))),
            cce_data_file_.get<h5::Dat>(detail::dataset_name_for_component(
                get<Tags::detail::InputDataSet<tag>>(dataset_names_))),
            computation_l_max, *time_span_start, *time_span_end,
            time_varies_fastest);
        cce_data_file_.close_current_object();
      });
  // the next time an update will be required
  return time_buffer_[std::min(*time_span_end - interpolator_length + 1,
                               time_buffer_.size() - 1)];
}

template <typename T>
std::unique_ptr<WorldtubeBufferUpdater<cce_metric_input_tags<T>>>
MetricWorldtubeH5BufferUpdater<T>::get_clone() const {
  return std::make_unique<MetricWorldtubeH5BufferUpdater>(
      MetricWorldtubeH5BufferUpdater{filename_});
}

template <typename T>
bool MetricWorldtubeH5BufferUpdater<T>::time_is_outside_range(
    const double time) const {
  return time < time_buffer_[0] or time > time_buffer_[time_buffer_.size() - 1];
}

template <typename T>
void MetricWorldtubeH5BufferUpdater<T>::pup(PUP::er& p) {
  p | time_buffer_;
  p | has_version_history_;
  p | filename_;
  p | descending_m_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

template <typename T>
void MetricWorldtubeH5BufferUpdater<T>::update_buffer(
    const gsl::not_null<T*> buffer_to_update, const h5::Dat& read_data,
    const size_t computation_l_max, const size_t time_span_start,
    const size_t time_span_end, const bool time_varies_fastest) const {
  const size_t number_of_columns = read_data.get_dimensions()[1];
  const size_t expected_size =
      (time_span_end - time_span_start) *
      (is_modal ? square(computation_l_max + 1)
                : Spectral::Swsh::number_of_swsh_collocation_points(
                      computation_l_max));
  // We require these to be the same since for nodal data we can't easily extend
  // nodal data like we can modal data.
  if (not is_modal and computation_l_max != l_max_) {
    ERROR(
        "When reading in nodal data, the LMax that "
        "MetricWorldtubeH5BufferUpdater was constructed with ("
        << l_max_
        << ") must be the same as the computation LMax passed to the "
           "update_buffer functions ("
        << computation_l_max << ").");
  }
  if (UNLIKELY(buffer_to_update->size() != expected_size)) {
    ERROR(
        "Incorrect " << (is_modal ? "modal" : "nodal")
                     << " storage size for the data to be loaded in. Expected "
                     << expected_size << " but got " << buffer_to_update->size()
                     << ". Time span is " << time_span_end << "-"
                     << time_span_start << "="
                     << time_span_end - time_span_start);
  }
  auto cols = alg::iota(std::vector<size_t>(number_of_columns - 1), 1_st);
  auto data_matrix =
      read_data.get_data_subset<std::vector<std::vector<double>>>(
          cols, time_span_start, time_span_end - time_span_start);

  *buffer_to_update = 0.0;
  for (size_t time_row = 0; time_row < time_span_end - time_span_start;
       ++time_row) {
    // If we have modal data, we must construct the ComplexModalVector
    if constexpr (is_modal) {
      for (int l = 0;
           l <= static_cast<int>(std::min(computation_l_max, l_max_)); ++l) {
        for (int m = -l; m <= l; ++m) {
          // -m because SpEC format is stored in decending m.
          const int em = descending_m_ ? -m : m;
          const size_t matrix_mode_index = Spectral::Swsh::goldberg_mode_index(
              l_max_, static_cast<size_t>(l), em);
          const size_t buffer_mode_index = Spectral::Swsh::goldberg_mode_index(
              computation_l_max, static_cast<size_t>(l), m);
          const size_t buffer_index =
              time_varies_fastest
                  ? buffer_mode_index * (time_span_end - time_span_start) +
                        time_row
                  : time_row * square(computation_l_max + 1) +
                        buffer_mode_index;

          (*buffer_to_update)[buffer_index] = std::complex<double>(
              data_matrix[time_row][2 * matrix_mode_index],
              data_matrix[time_row][2 * matrix_mode_index + 1]);
        }
      }
    } else {
      // Otherwise we just read the nodal data in as is
      for (size_t i = 0; i < cols.size(); i++) {
        const size_t buffer_index =
            time_varies_fastest
                ? i * (time_span_end - time_span_start) + time_row
                : time_row * cols.size() + i;
        (*buffer_to_update)[buffer_index] = data_matrix[time_row][i];
      }
    }
  }
}

namespace {
// Convenience metafunction to return the correct tag depending on the template
// parameter T
template <typename T, typename Tag>
struct name_tag {
  using type = Tags::detail::InputDataSet<tmpl::conditional_t<
      std::is_same_v<T, ComplexModalVector>,
      Spectral::Swsh::Tags::SwshTransform<Tag>, Tags::BoundaryValue<Tag>>>;
};

template <typename T, typename Tag>
using name_tag_t = typename name_tag<T, Tag>::type;
}  // namespace

template <typename T>
BondiWorldtubeH5BufferUpdater<T>::BondiWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename,
    const std::optional<double> extraction_radius)
    : cce_data_file_{cce_data_filename}, filename_{cce_data_filename} {
  get<name_tag_t<T, Tags::BondiBeta>>(dataset_names_) = "Beta";
  get<name_tag_t<T, Tags::BondiU>>(dataset_names_) = "U";
  get<name_tag_t<T, Tags::BondiQ>>(dataset_names_) = "Q";
  get<name_tag_t<T, Tags::BondiW>>(dataset_names_) = "W";
  get<name_tag_t<T, Tags::BondiJ>>(dataset_names_) = "J";
  get<name_tag_t<T, Tags::Dr<Tags::BondiJ>>>(dataset_names_) = "DrJ";
  get<name_tag_t<T, Tags::Du<Tags::BondiJ>>>(dataset_names_) = "H";
  get<name_tag_t<T, Tags::BondiR>>(dataset_names_) = "R";
  get<name_tag_t<T, Tags::Du<Tags::BondiR>>>(dataset_names_) = "DuR";

  // the extraction radius is typically not used in the Bondi system, so we
  // don't error if it isn't parsed from the filename. Instead, we'll just error
  // if the invalid extraction radius value is ever retrieved using
  // `get_extraction_radius`.
  extraction_radius_ =
      Cce::get_extraction_radius(cce_data_filename, extraction_radius, false);

  detail::set_time_buffer_and_lmax(make_not_null(&time_buffer_), l_max_,
                                   cce_data_file_.get<h5::Dat>("/U"), false,
                                   is_modal, true);
  cce_data_file_.close_current_object();
}

template <typename T>
double BondiWorldtubeH5BufferUpdater<T>::update_buffers_for_time(
    const gsl::not_null<Variables<tags_for_writing>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth, const bool time_varies_fastest) const {
  return detail::update_buffers_for_time<is_modal, tags_for_writing>(
      buffers, time_span_start, time_span_end, time, computation_l_max, l_max_,
      interpolator_length, buffer_depth, time_buffer_, dataset_names_,
      cce_data_file_, time_varies_fastest);
}

template <typename T>
void BondiWorldtubeH5BufferUpdater<T>::pup(PUP::er& p) {
  p | time_buffer_;
  p | filename_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

KleinGordonWorldtubeH5BufferUpdater::KleinGordonWorldtubeH5BufferUpdater(
    const std::string& cce_data_filename,
    const std::optional<double> extraction_radius)
    : cce_data_file_{cce_data_filename}, filename_{cce_data_filename} {
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::KleinGordonPsi>>>(
      dataset_names_) = "KGPsi";
  get<Tags::detail::InputDataSet<
      Spectral::Swsh::Tags::SwshTransform<Tags::KleinGordonPi>>>(
      dataset_names_) = "dtKGPsi";

  // the extraction radius is typically not used in the Klein-Gordon system, so
  // we don't error if it isn't parsed from the filename. Instead, we'll just
  // error if the invalid extraction radius value is ever retrieved using
  // `get_extraction_radius`.
  extraction_radius_ =
      Cce::get_extraction_radius(cce_data_filename, extraction_radius, false);

  detail::set_time_buffer_and_lmax(make_not_null(&time_buffer_), l_max_,
                                   cce_data_file_.get<h5::Dat>("/KGPsi"), true,
                                   true, true);
  cce_data_file_.close_current_object();
}

double KleinGordonWorldtubeH5BufferUpdater::update_buffers_for_time(
    const gsl::not_null<Variables<klein_gordon_input_tags>*> buffers,
    const gsl::not_null<size_t*> time_span_start,
    const gsl::not_null<size_t*> time_span_end, const double time,
    const size_t computation_l_max, const size_t interpolator_length,
    const size_t buffer_depth, const bool time_varies_fastest) const {
  if (UNLIKELY(not time_varies_fastest)) {
    ERROR(
        "KleinGordon worldtube data can only be read from H5 in "
        "time-varies-fastest form.");
  }
  return detail::update_buffers_for_time<true, klein_gordon_input_tags>(
      buffers, time_span_start, time_span_end, time, computation_l_max, l_max_,
      interpolator_length, buffer_depth, time_buffer_, dataset_names_,
      cce_data_file_);
}

void KleinGordonWorldtubeH5BufferUpdater::pup(PUP::er& p) {
  p | time_buffer_;
  p | filename_;
  p | l_max_;
  p | extraction_radius_;
  p | dataset_names_;
  if (p.isUnpacking()) {
    cce_data_file_ = h5::H5File<h5::AccessType::ReadOnly>{filename_};
  }
}

template <typename T>
PUP::able::PUP_ID MetricWorldtubeH5BufferUpdater<T>::my_PUP_ID = 0;  // NOLINT
template <typename T>
PUP::able::PUP_ID BondiWorldtubeH5BufferUpdater<T>::my_PUP_ID = 0;     // NOLINT
PUP::able::PUP_ID KleinGordonWorldtubeH5BufferUpdater::my_PUP_ID = 0;  // NOLINT

template class MetricWorldtubeH5BufferUpdater<ComplexModalVector>;
template class MetricWorldtubeH5BufferUpdater<DataVector>;
template class BondiWorldtubeH5BufferUpdater<ComplexModalVector>;
template class BondiWorldtubeH5BufferUpdater<ComplexDataVector>;

#define SPIN(data) BOOST_PP_TUPLE_ELEM(0, data)
#define IS_MODAL(data) BOOST_PP_TUPLE_ELEM(1, data)
#define VEC_TYPE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                       \
  template void detail::update_buffer<IS_MODAL(data), SPIN(data)>( \
      const gsl::not_null<VEC_TYPE(data)*> buffer_to_update,       \
      const h5::Dat& read_data, const size_t computation_l_max,    \
      const size_t l_max, const size_t time_span_start,            \
      const size_t time_span_end, const bool time_varies_fastest);

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2), (true, false),
                        (ComplexModalVector, ComplexDataVector))

#undef INSTANTIATE
#undef VEC_TYPE
#undef IS_MODAL
#undef SPIN
}  // namespace Cce
