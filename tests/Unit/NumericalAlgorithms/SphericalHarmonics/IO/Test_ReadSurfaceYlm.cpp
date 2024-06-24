// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "ParallelAlgorithms/Interpolation/Callbacks/ObserveSurfaceData.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"

namespace {
const size_t l_max_column_number = 4;

// generate test Strahlkorpers with a random radius function and given l_maxes
// and expansion center
template <typename Frame, size_t NumTimes>
std::array<ylm::Strahlkorper<Frame>, NumTimes> generate_test_strahlkorpers(
    const std::array<double, 3> expansion_center,
    const std::array<size_t, NumTimes> l_maxes) {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.1, 2.0);

  std::array<ylm::Strahlkorper<Frame>, NumTimes> strahlkorpers{};
  for (size_t i = 0; i < NumTimes; i++) {
    const auto radius = make_with_random_values<DataVector>(
        make_not_null(&generator), distribution,
        DataVector(ylm::Spherepack::physical_size(gsl::at(l_maxes, i),
                                                  gsl::at(l_maxes, i)),
                   std::numeric_limits<double>::signaling_NaN()));
    gsl::at(strahlkorpers, i) = ylm::Strahlkorper<Frame>(
        gsl::at(l_maxes, i), gsl::at(l_maxes, i), radius, expansion_center);
  }

  return strahlkorpers;
}

// Generate the data to be written using the helper function for
// ::intrp::callbacks::ObserveSurfaceData instead of spinning up the Action
// Testing Framework (AFT) to write it since the helper function handles all
// of the legend and data building logic and ObserveSurfaceData simply writes
// what it generates
template <typename Frame, size_t NumTimes>
void write_test_strahlkorpers(
    const std::array<ylm::Strahlkorper<Frame>, NumTimes>& strahlkorpers,
    const std::string& test_filename, const std::string& subfile_name,
    const std::array<double, NumTimes> times, const size_t max_l) {
  std::vector<std::string> legend{};
  std::vector<std::vector<double>> data(NumTimes);
  for (size_t i = 0; i < NumTimes; i++) {
    legend.resize(0);  // clear and reuse for next row of data
    ylm::fill_ylm_legend_and_data(
        make_not_null(&legend), make_not_null(&(data[i])),
        gsl::at(strahlkorpers, i), gsl::at(times, i), max_l);
  }

  h5::H5File<h5::AccessType::ReadWrite> test_file(test_filename, true);
  auto& file = test_file.insert<h5::Dat>("/" + subfile_name, legend);
  file.append(data);
  test_file.close_current_object();
}

// test reading in the n last times of data expected to have been written
// (expected_strahlkorpers), where n = num_times_requested
template <typename Frame, size_t NumTimes>
void check_read_ylm_data(const std::string& test_filename,
                         const std::string& surface_subfile_name,
                         const size_t num_times_requested,
                         const std::array<ylm::Strahlkorper<Frame>, NumTimes>&
                             expected_strahlkorpers) {
  ASSERT(
      NumTimes >= num_times_requested,
      "Requesting to read more rows of Ylm test data than the total number of "
      "rows expected to be written and therefore able to be read.");

  const std::vector<ylm::Strahlkorper<Frame>> strahlkorpers =
      ylm::read_surface_ylm<Frame>(test_filename, surface_subfile_name,
                                   num_times_requested);
  CHECK(strahlkorpers.size() == num_times_requested);

  // check n last times where n = num_times_requested
  for (size_t i = 0, expected_row_number = NumTimes - num_times_requested;
       i < num_times_requested; i++, expected_row_number++) {
    const auto& expected_strahlkorper =
        gsl::at(expected_strahlkorpers, expected_row_number);
    const std::array<double, 3> expected_expansion_center =
        expected_strahlkorper.expansion_center();
    const size_t expected_l_max = expected_strahlkorper.l_max();
    const DataVector& expected_spectral_coefficients =
        expected_strahlkorper.coefficients();
    const size_t expected_spectral_size = expected_spectral_coefficients.size();

    const auto& strahlkorper = strahlkorpers[i];
    CHECK(strahlkorper.expansion_center() == expected_expansion_center);
    CHECK(strahlkorper.l_max() == expected_l_max);
    CHECK(strahlkorper.coefficients().size() == expected_spectral_size);
    CHECK(strahlkorper.coefficients() == expected_spectral_coefficients);
  }
}

// Write and read in a file containing a legend or data that is expected to
// generate an error upon attempting to read
template <typename Frame>
void write_error_file_and_try_to_read(
    const std::string& filename, const std::string& subfile_name,
    const std::vector<std::string>& legend,
    const std::vector<std::vector<double>>& data,
    const size_t num_times_to_read) {
  h5::H5File<h5::AccessType::ReadWrite> test_file{filename, true};
  auto& file = test_file.insert<h5::Dat>("/" + subfile_name, legend);
  file.append(data);
  test_file.close_current_object();

  const auto strahlkorpers =
      ylm::read_surface_ylm<Frame>(filename, subfile_name, num_times_to_read);
}

void test_errors() {
  const std::vector<std::string> ylm_legend_without_coefs{
      "Time", "InertialExpansionCenter_x", "InertialExpansionCenter_y",
      "InertialExpansionCenter_z", "Lmax"};

  const std::vector<std::string> l_max_3_coef_headers = {
      "coef(0,0)",  "coef(1,-1)", "coef(1,0)",  "coef(1,1)",
      "coef(2,-2)", "coef(2,-1)", "coef(2,0)",  "coef(2,1)",
      "coef(2,2)",  "coef(3,-3)", "coef(3,-2)", "coef(3,-1)",
      "coef(3,0)",  "coef(3,1)",  "coef(3,2)",  "coef(3,3)"};

  // construct a legend that is properly formatted
  std::vector<std::string> good_legend = ylm_legend_without_coefs;
  good_legend.insert(good_legend.end(), l_max_3_coef_headers.begin(),
                     l_max_3_coef_headers.end());

  // construct data that is properly formatted
  const std::array<double, 3> times{{0.1, 0.2, 0.3}};
  const std::array<double, 3> l_maxes{{2.0, 3.0, 2.0}};
  std::vector<std::vector<double>> good_data{
      times.size(), std::vector<double>(good_legend.size(), 0.0)};
  good_data[0][l_max_column_number] = gsl::at(l_maxes, 0);
  good_data[1][l_max_column_number] = gsl::at(l_maxes, 0);
  good_data[2][l_max_column_number] = gsl::at(l_maxes, 0);

  const std::string filename{"TestReadYlmErrors.h5"};

  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }

  // expansion center legend errors

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      std::vector<std::string> bad_legend = good_legend;
                      bad_legend[2] = "InertialExpansionCenter_z";

                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "WrongCoord", bad_legend, good_data, 3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "In column 2 of the Ylm legend, expected header "
                        "InertialExpansionCenter_y but got header "
                        "InertialExpansionCenter_z"));

  CHECK_THROWS_WITH(
      ([&filename, &good_legend, &good_data]() {
        std::vector<std::string> bad_legend = good_legend;
        bad_legend[3] = "ExpansionCenter_z";

        write_error_file_and_try_to_read<Frame::Inertial>(
            filename, "NoFrame", bad_legend, good_data, 3);
      }()),
      Catch::Matchers::ContainsSubstring(
          "In column 3 of the Ylm legend, expected header "
          "InertialExpansionCenter_z but got header ExpansionCenter_z"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      write_error_file_and_try_to_read<Frame::Grid>(
                          filename, "WrongFrame", good_legend, good_data, 3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "In column 1 of the Ylm legend, expected header "
                        "GridExpansionCenter_x but got header "
                        "InertialExpansionCenter_x"));

  // coefficient legend errors

  CHECK_THROWS_WITH(
      ([&filename, &good_legend, &good_data]() {
        std::vector<std::string> bad_legend = good_legend;
        bad_legend[8] = "1, 1";

        write_error_file_and_try_to_read<Frame::Inertial>(
            filename, "WrongCoefFormat", bad_legend, good_data, 3);
      }()),
      Catch::Matchers::ContainsSubstring(
          "In column 8 of the Ylm legend, expected header coef(1,1) but got "
          "header 1, 1"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      std::vector<std::string> bad_legend = good_legend;
                      bad_legend[12] = "coef(2,2)";

                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "WrongCoefOrder", bad_legend, good_data, 3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "In column 12 of the Ylm legend, expected header "
                        "coef(2,1) but got header coef(2,2)"));

  // data errors

  CHECK_THROWS_WITH(([&filename, &good_legend]() {
                      const std::vector<std::vector<double>> bad_data{};
                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "EmptyData", good_legend, bad_data, 1);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "The Ylm data to read from contain 0 rows"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "NotEnoughTimesWritten", good_legend,
                          good_data, 4);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "The requested number of time values (4) is more than "
                        "the number of rows in the Ylm data"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      std::vector<std::vector<double>> bad_data = good_data;
                      bad_data[0][l_max_column_number] = 1.2;

                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "NonIntegralLmax", good_legend, bad_data,
                          3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "Row 0 of the Ylm data has an invalid Lmax value"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      std::vector<std::vector<double>> bad_data = good_data;
                      bad_data[1][l_max_column_number] = -1.0;

                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "NegativeLmax", good_legend, bad_data, 3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "Row 1 of the Ylm data has an invalid Lmax value"));

  CHECK_THROWS_WITH(([&filename, &good_legend, &good_data]() {
                      std::vector<std::vector<double>> bad_data = good_data;
                      // set an Lmax that requires more columns of data than
                      // given, i.e. `good_legend` only has room for coefs for
                      // l <= 3
                      bad_data[2][l_max_column_number] = 4;

                      write_error_file_and_try_to_read<Frame::Inertial>(
                          filename, "NotEnoughColumnsForLmax", good_legend,
                          bad_data, 3);
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "Row 2 of the Ylm data does not have enough "
                        "coefficients for the Lmax. For Lmax = 4, expected at "
                        "least 25 coefficient columns and 30 total columns"));

  CHECK_THROWS_WITH(
      ([&filename, &good_legend, &good_data]() {
        std::vector<std::vector<double>> bad_data = good_data;
        bad_data[0].back() = 0.8;

        write_error_file_and_try_to_read<Frame::Inertial>(
            filename, "NonZeroHigherCoefs", good_legend, bad_data, 3);
      }()),
      Catch::Matchers::ContainsSubstring(
          "Row 0 of the Ylm data has Lmax = 2 but non-zero coefficients for "
          "l > Lmax"));
  if (file_system::check_if_file_exists(filename)) {
    file_system::rm(filename, true);
  }
}

void test_single_time(const std::string& test_filename) {
  INFO("Test Single Time");

  using frame = Frame::Inertial;
  const std::string subfile_name = "SurfaceA_SingleTime_Ylm";
  const std::array<double, 3> expansion_center{{-0.5, -0.1, 0.3}};
  const size_t max_l = 3;

  constexpr size_t number_of_times = 3;
  const std::array<double, number_of_times> written_times{{0.0, 0.1, 0.2}};
  const std::array<size_t, number_of_times> l_maxes{{3, 3, 3}};

  const auto expected_strahlkorpers =
      generate_test_strahlkorpers<frame>(expansion_center, l_maxes);
  write_test_strahlkorpers(expected_strahlkorpers, test_filename, subfile_name,
                           written_times, max_l);

  {
    const ylm::Strahlkorper<frame> read_in_strahlkorper =
        ylm::read_surface_ylm_single_time<frame>(test_filename, subfile_name,
                                                 0.1, 1e-12);

    CHECK(expected_strahlkorpers[1] == read_in_strahlkorper);
  }

  {
    const ylm::Strahlkorper<frame> read_in_strahlkorper =
        ylm::read_surface_ylm_single_time<frame>(test_filename, subfile_name,
                                                 0.2, 1e-2);

    CHECK(expected_strahlkorpers[2] == read_in_strahlkorper);
  }

  {
    const ylm::Strahlkorper<Frame::Grid> read_in_strahlkorper =
        ylm::read_surface_ylm_single_time<Frame::Grid>(
            test_filename, subfile_name, 0.2, 1e-2, false);

    // Just check the coefficients since == won't work with different frames
    CHECK(expected_strahlkorpers[2].coefficients() ==
          read_in_strahlkorper.coefficients());
  }

  CHECK_THROWS_WITH(
      ylm::read_surface_ylm_single_time<frame>(test_filename, subfile_name, 0.3,
                                               1e-12),
      Catch::Matchers::ContainsSubstring("Could not find time") and
          Catch::Matchers::ContainsSubstring("Available times are:"));

  CHECK_THROWS_WITH(ylm::read_surface_ylm_single_time<frame>(
                        test_filename, subfile_name, 0.2, 1.0),
                    Catch::Matchers::ContainsSubstring(
                        "Found more than one time in the subfile") and
                        Catch::Matchers::ContainsSubstring(
                            "that is within a relative_epsilon of"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.SphericalHarmonics.ReadSurfaceYlm",
                  "[NumericalAlgorithms][Unit]") {
  test_errors();

  // Temporary file with test data to read in
  const std::string test_filename{"TestReadYlm.h5"};
  if (file_system::check_if_file_exists(test_filename)) {
    file_system::rm(test_filename, true);
  }

  {
    INFO("Test A");

    // first test surface
    using frame_a = Frame::Grid;
    const std::string subfile_name_a = "SurfaceA_Ylm";
    const std::array<double, 3> expansion_center_a{{-0.5, -0.1, 0.3}};
    const size_t max_l_a = 3;

    constexpr size_t number_of_times_a = 3;
    const std::array<double, number_of_times_a> written_times_a{
        {0.0, 0.1, 0.2}};
    // test that a mix of different l_max values can be read from the same
    // subfile
    const std::array<size_t, number_of_times_a> l_maxes_a{{3, 2, 3}};

    const auto strahlkorpers_a =
        generate_test_strahlkorpers<frame_a>(expansion_center_a, l_maxes_a);
    write_test_strahlkorpers(strahlkorpers_a, test_filename, subfile_name_a,
                             written_times_a, max_l_a);

    check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 1,
                                 strahlkorpers_a);
    check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 2,
                                 strahlkorpers_a);
    check_read_ylm_data<frame_a>(test_filename, subfile_name_a, 3,
                                 strahlkorpers_a);
  }

  {
    INFO("Test B");

    // second test surface
    using frame_b = Frame::Distorted;
    const std::string subfile_name_b = "SurfaceB";
    const std::array<double, 3> expansion_center_b{{0.0, 0.2, -0.6}};
    const size_t max_l_b = 4;

    // edge case: only one row written
    constexpr size_t number_of_times_b = 1;
    const std::array<double, number_of_times_b> written_times_b{{0.7}};
    const std::array<size_t, number_of_times_b> l_maxes_b{{3}};

    const auto strahlkorpers_b =
        generate_test_strahlkorpers<frame_b>(expansion_center_b, l_maxes_b);
    write_test_strahlkorpers(strahlkorpers_b, test_filename, subfile_name_b,
                             written_times_b, max_l_b);

    check_read_ylm_data<frame_b>(test_filename, subfile_name_b, 1,
                                 strahlkorpers_b);
  }

  test_single_time(test_filename);

  // Delete the temporary file created for this test
  file_system::rm(test_filename, true);
}
