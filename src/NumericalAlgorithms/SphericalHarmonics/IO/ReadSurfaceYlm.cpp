// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"

namespace ylm {
namespace {
const size_t num_non_coef_headers = 5;

// check that the headers of the legend are what we expect them to be
template <typename Frame>
void check_legend(const std::vector<std::string>& legend,
                  const bool check_frame = true) {
  const std::string expected_frame{get_output(Frame{})};

  // check format and ordering of non-coefficient headers
  const auto check_header = [&legend](const size_t column,
                                      const std::string& expected_header,
                                      const bool check = true) {
    const std::string& read_header = gsl::at(legend, column);
    if (check and read_header != expected_header) {
      ERROR("In column " << column << " of the Ylm legend, expected header "
                         << expected_header << " but got header " << read_header
                         << ".");
    }
  };
  check_header(0, "Time");
  check_header(1, expected_frame + "ExpansionCenter_x", check_frame);
  check_header(2, expected_frame + "ExpansionCenter_y", check_frame);
  check_header(3, expected_frame + "ExpansionCenter_z", check_frame);
  check_header(4, "Lmax");

  // check format and ordering of coefficient headers
  int expected_l = 0;
  int expected_m = 0;
  size_t coef_column_number = num_non_coef_headers;
  while (coef_column_number < legend.size()) {
    const std::string& coefficient_header = legend[coef_column_number];
    const std::string expected_coefficient_header = {
        MakeString{} << "coef(" << std::to_string(expected_l) << ","
                     << std::to_string(expected_m) << ")"};
    if (coefficient_header != expected_coefficient_header) {
      ERROR(MakeString{}
            << "In column " << coef_column_number
            << " of the Ylm legend, expected header "
            << expected_coefficient_header << " but got header "
            << coefficient_header
            << ".\n\n Coefficient headers are expected to take the form "
               "coef(l,m) and be ordered first by ascending l and then for "
               "each l, ordered by ascending m, i.e. coef(0,0), coef(1,-1), "
               "coef(1,0), coef(1,1), coef(2,-2), ...");
    }

    if (expected_l == expected_m) {
      expected_l++;
      expected_m = -expected_l;
    } else {
      expected_m++;
    }

    coef_column_number++;
  }
}

// checks if a double is a nonnegative integer
bool is_nonnegative_int(const double n) {
  return n == abs(n) and n == floor(n);
}

template <typename Frame>
Strahlkorper<Frame> read_surface_ylm_row(const Matrix& ylm_data,
                                         const size_t row_number) {
  const size_t l_max_column_number = 4;
  const std::array<double, 3> expansion_center{{ylm_data(row_number, 1),
                                                ylm_data(row_number, 2),
                                                ylm_data(row_number, 3)}};
  const double l_max_from_file = ylm_data(row_number, l_max_column_number);
  if (not is_nonnegative_int(l_max_from_file)) {
    ERROR("Row " << row_number << " of the Ylm data has an invalid Lmax value ("
                 << l_max_from_file << ") in column " << l_max_column_number
                 << ". The value of Lmax should be a nonnegative integer.");
  }

  const auto l_max = static_cast<size_t>(l_max_from_file);
  // l_max == m_max
  const size_t spectral_size = Spherepack::spectral_size(l_max, l_max);
  // We only write and store half of the coefficients. This is the minimum
  // number of coefficients needed to describe the strahlkorper given the l_max,
  // i.e. columns of 0.0 for higher coefficients are okay
  const size_t min_expected_num_coefficients = spectral_size / 2;
  const size_t min_expected_num_columns =
      min_expected_num_coefficients + num_non_coef_headers;
  const size_t actual_num_columns = ylm_data.columns();

  if (actual_num_columns < min_expected_num_columns) {
    ERROR("Row " << row_number
                 << " of the Ylm data does not have enough coefficients for "
                    "the Lmax. For Lmax = "
                 << l_max << ", expected at least "
                 << min_expected_num_coefficients << " coefficient columns and "
                 << min_expected_num_columns << " total columns.");
  }

  ModalVector spectral_coefficients(spectral_size, 0.0);
  size_t coef_column_number = num_non_coef_headers;
  SpherepackIterator iter(l_max, l_max);
  // read in expected coefficients for the given l_max
  for (size_t l = 0; l <= l_max; l++) {
    for (int m = -static_cast<int>(l); m <= static_cast<int>(l); m++) {
      iter.set(l, m);

      const double coefficient = ylm_data(row_number, coef_column_number);
      spectral_coefficients[iter()] = coefficient;

      coef_column_number++;
    }
  }
  // make sure any higher order coefficients (l > l_max) present in the data
  // that was read in are 0.0
  while (coef_column_number < actual_num_columns) {
    if (ylm_data(row_number, coef_column_number) != 0.0) {
      ERROR("Row " << row_number << " of the Ylm data has Lmax = " << l_max
                   << " but non-zero coefficients for l > Lmax.");
    }
    coef_column_number++;
  }

  Strahlkorper<Frame> strahlkorper(l_max, l_max, spectral_coefficients,
                                   expansion_center);

  return strahlkorper;
}
}  // namespace

template <typename Frame>
ylm::Strahlkorper<Frame> read_surface_ylm_single_time(
    const std::string& file_name, const std::string& surface_subfile_name,
    const double time, const double relative_epsilon, const bool check_frame) {
  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  const std::string ylm_subfile_name{std::string{"/"} + surface_subfile_name};
  const auto& ylm_file = file.get<h5::Dat>(ylm_subfile_name);
  const auto& ylm_legend = ylm_file.get_legend();
  check_legend<Frame>(ylm_legend, check_frame);

  std::vector<size_t> columns(ylm_legend.size());
  std::iota(std::begin(columns), std::end(columns), 0);

  const auto& dimensions = ylm_file.get_dimensions();
  const auto num_rows = static_cast<size_t>(dimensions[0]);
  const Matrix times =
      ylm_file.get_data_subset(std::vector<size_t>{0_st}, 0, num_rows);

  std::optional<size_t> row_number{};
  for (size_t i = 0; i < num_rows; i++) {
    if (equal_within_roundoff(time, times(i, 0), relative_epsilon, time)) {
      if (row_number.has_value()) {
        ERROR("Found more than one time in the subfile "
              << surface_subfile_name << " of the H5 file " << file_name
              << " that is within a relative_epsilon of " << relative_epsilon
              << " of the time requested " << time);
      }

      row_number = i;
    }
  }

  if (not row_number.has_value()) {
    ERROR("Could not find time " << time << " in subfile "
                                 << surface_subfile_name << " of H5 file "
                                 << file_name << ". Available times are:\n"
                                 << times);
  }

  const Matrix data = ylm_file.get_data_subset(columns, row_number.value());

  // Zero for row number because this matrix only has one row
  return read_surface_ylm_row<Frame>(data, 0);
}

template <typename Frame>
std::vector<Strahlkorper<Frame>> read_surface_ylm(
    const std::string& file_name, const std::string& surface_subfile_name,
    const size_t requested_number_of_times_from_end) {
  ASSERT(requested_number_of_times_from_end > 0,
         "Must request to read in at least one row (time) of Ylm data.");

  h5::H5File<h5::AccessType::ReadOnly> file{file_name};
  const std::string ylm_subfile_name{std::string{"/"} + surface_subfile_name};
  const auto& ylm_file = file.get<h5::Dat>(ylm_subfile_name);

  // number of rows available to read in
  const size_t total_number_of_times = gsl::at(ylm_file.get_dimensions(), 0);
  if (total_number_of_times == 0) {
    ERROR("The Ylm data to read from contain 0 rows (times) of data.");
  }

  if (requested_number_of_times_from_end > total_number_of_times) {
    ERROR("The requested number of time values ("
          << requested_number_of_times_from_end
          << ") is more than the number of rows in the Ylm data that was read "
             "in ("
          << total_number_of_times << ")");
  }

  const auto& ylm_legend = ylm_file.get_legend();
  check_legend<Frame>(ylm_legend);

  std::vector<size_t> columns(ylm_legend.size());
  std::iota(std::begin(columns), std::end(columns), 0);
  // grab all columns of the last requested_number_of_times_from_end rows
  const auto ylm_data_subset = ylm_file.get_data_subset(
      columns, total_number_of_times - requested_number_of_times_from_end,
      requested_number_of_times_from_end);

  std::vector<Strahlkorper<Frame>> strahlkorpers(
      requested_number_of_times_from_end);
  for (size_t i = 0; i < requested_number_of_times_from_end; i++) {
    strahlkorpers[i] = read_surface_ylm_row<Frame>(ylm_data_subset, i);
  }

  file.close_current_object();
  return strahlkorpers;
}
}  // namespace ylm

#define FRAMETYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template std::vector<ylm::Strahlkorper<FRAMETYPE(data)>>                   \
  ylm::read_surface_ylm<>(const std::string& file_name,                      \
                          const std::string& surface_subfile_name,           \
                          size_t requested_number_of_times_from_end);        \
  template ylm::Strahlkorper<FRAMETYPE(data)>                                \
  ylm::read_surface_ylm_single_time<>(                                       \
      const std::string& file_name, const std::string& surface_subfile_name, \
      double time, double relative_epsilon, bool check_frame);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Inertial, Frame::Distorted))

#undef INSTANTIATE
#undef FRAMETYPE
