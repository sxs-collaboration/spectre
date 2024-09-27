// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename T>
T expected_data(const std::vector<double>& unformatted_data,
                const std::array<size_t, 2>& size) {
  REQUIRE(size[0] * size[1] == unformatted_data.size());
  T result{};

  if constexpr (std::is_same_v<T, Matrix>) {
    result = Matrix(size[0], size[1]);
  } else {
    result = std::vector<std::vector<double>>(size[0]);
    for (size_t i = 0; i < size[0]; i++) {
      result[i].resize(size[1]);
    }
  }

  for (size_t i = 0; i < size[0]; i++) {
    for (size_t j = 0; j < size[1]; j++) {
      if constexpr (std::is_same_v<T, Matrix>) {
        result(i, j) = unformatted_data[j + i * size[1]];
      } else {
        result[i][j] = unformatted_data[j + i * size[1]];
      }
    }
  }

  return result;
}

void check_get_data(const h5::Dat& dat_file,
                    const std::vector<double>& unformatted_data,
                    const std::array<size_t, 2>& size) {
  {
    using T = Matrix;
    const auto data = expected_data<T>(unformatted_data, size);
    CHECK(dat_file.get_data<T>() == data);
  }
  {
    using T = std::vector<std::vector<double>>;
    const auto data = expected_data<T>(unformatted_data, size);
    CHECK(dat_file.get_data<T>() == data);
  }
}

void check_get_data_subset(const h5::Dat& dat_file,
                           const std::vector<double>& unformatted_data,
                           const std::array<size_t, 2>& size,
                           const std::vector<size_t>& these_columns,
                           size_t first_row, size_t num_rows) {
  {
    using T = Matrix;
    const auto data = expected_data<T>(unformatted_data, size);
    CHECK(dat_file.get_data_subset<T>(these_columns, first_row, num_rows) ==
          data);
  }
  {
    using T = std::vector<std::vector<double>>;
    const auto data = expected_data<T>(unformatted_data, size);
    CHECK(dat_file.get_data_subset<T>(these_columns, first_row, num_rows) ==
          data);
  }
}

void test_errors() {
  std::string file_name{"./Unit.IO.H5.Dat.FileErrorObjectAlreadyExists.h5"};
  CHECK_THROWS_WITH(
      ([&file_name]() {
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        const std::vector<std::string> legend{"Time", "Error L2", "Error L1",
                                              "Error"};
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& error_file =
              my_file.insert<h5::Dat>("/L2_errors///", legend, version_number);
          error_file.append(std::vector<double>{0, 0.1, 0.2, 0.3});
          my_file.close_current_object();
        }
        { my_file.insert<h5::Dat>("/L2_errors//", legend, version_number); }
      }()),
      Catch::Matchers::ContainsSubstring(
          "Cannot insert an Object that already exists. Failed to "
          "add Object named: /L2_errors"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.Dat.FileErrorObjectAlreadyOpenGet.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const std::vector<std::string> legend{"DummyLegend"};
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& dat_file = my_file.insert<h5::Dat>("/DummyPath", legend);
        auto& get_dat_file = my_file.get<h5::Dat>("/DummyPath");
        (void)dat_file;
        (void)get_dat_file;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot open object /DummyPath."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.Dat.FileErrorObjectAlreadyOpenInsert.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const std::vector<std::string> legend{"DummyLegend"};
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& dat_file = my_file.insert<h5::Dat>("/DummyPath", legend);
        auto& dat_file2 = my_file.insert<h5::Dat>("/DummyPath2");
        (void)dat_file;
        (void)dat_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot insert object /DummyPath2."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.Dat.FileErrorObjectAlreadyOpenTryInsert.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const std::vector<std::string> legend{"DummyLegend"};
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& dat_file = my_file.insert<h5::Dat>("/DummyPath", legend);
        auto& dat_file2 = my_file.try_insert<h5::Dat>("/DummyPath2");
        (void)dat_file;
        (void)dat_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot try to insert "
          "object /DummyPath2."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.Dat.FileErrorObjectAlreadyOpen.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const std::vector<std::string> legend{"DummyLegend"};
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& dat_file = my_file.insert<h5::Dat>("/DummyPath", legend);
          my_file.close_current_object();
          (void)dat_file;
          auto& dat_file2 = my_file.insert<h5::Dat>("/Dummy/Path2", legend);
          my_file.close_current_object();
          (void)dat_file2;
        }
        auto& dat_file2 = my_file.get<h5::Dat>("/Dummy/Path2");
        auto& dat_file = my_file.get<h5::Dat>("/DummyPath");
        (void)dat_file;
        (void)dat_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /Dummy/Path2 already open. Cannot open object /DummyPath."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
}

void test_core_functionality() {
  const std::string h5_file_name("Unit.IO.H5.Dat.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  std::vector<std::string> legend{"Time", "Error L2", "Error L1", "Error"};

  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  my_file.insert<h5::Dat>("/L2_errors", legend, version_number);
  my_file.close_current_object();

  // Check that the Dat file is found to be a subgroup of the file
  const auto groups_in_file = my_file.groups();
  CHECK(alg::find(groups_in_file, std::string{"L2_errors.dat"}) !=
        groups_in_file.end());

  // pass bad values to make sure the original ones aren't overridden.
  auto evil_legend = legend;
  evil_legend.emplace_back("Uh oh");
  auto& error_file =
      my_file.try_insert<h5::Dat>("/L2_errors", evil_legend, version_number);

  error_file.append(std::vector<std::vector<double>>{{0.00, 0.10, 0.20, 0.30},
                                                     {0.11, 0.40, 0.50, 0.60},
                                                     {0.22, 0.55, 0.60, 0.80}});
  error_file.append(std::vector<std::vector<double>>{});
  error_file.append(std::vector<double>{0.33, 0.66, 0.77, 0.90});

  // Check version info is correctly retrieved from Dat file
  CHECK(error_file.get_version() == version_number);

  // Check getting the header from Dat file
  std::stringstream ss;
  ss << "# ";
  auto build_info = info_from_build();
  ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
  const auto& header = error_file.get_header();
  CHECK("#\n# File created on "s ==
        header.substr(0, header.find("File created on ") + 16));
  CHECK(ss.str() == header.substr(header.find("# SpECTRE")));

  CHECK(error_file.get_legend() == legend);

  // Check data is retrieved correctly from Dat file
  std::array<hsize_t, 2> size_of_data{{4, 4}};
  CHECK(error_file.get_dimensions() == size_of_data);

  // For the docs
  {
    // [h5dat_get_data]
    const Matrix matrix_data_in_dat_file = error_file.get_data();
    const auto matrix_data_in_dat_file2 = error_file.get_data<Matrix>();
    const auto vector_data_in_dat_file =
        error_file.get_data<std::vector<std::vector<double>>>();
    // [h5dat_get_data]
    // [h5dat_get_subset]
    const Matrix matrix_subset_data_in_dat_file =
        error_file.get_data_subset({1, 3}, 1, 3);
    const auto matrix_subset_data_in_dat_file2 =
        error_file.get_data_subset<Matrix>({1, 3}, 1, 3);
    const auto vector_subset_data_in_dat_file =
        error_file.get_data_subset<std::vector<std::vector<double>>>({1, 3}, 1,
                                                                     3);
    // [h5dat_get_subset]
    (void)matrix_data_in_dat_file;
    (void)matrix_data_in_dat_file2;
    (void)vector_data_in_dat_file;
    (void)matrix_subset_data_in_dat_file;
    (void)matrix_subset_data_in_dat_file2;
    (void)vector_subset_data_in_dat_file;
  }

  check_get_data(error_file,
                 {0.0, 0.1, 0.2, 0.3, 0.11, 0.4, 0.5, 0.6, 0.22, 0.55, 0.6, 0.8,
                  0.33, 0.66, 0.77, 0.9},
                 {4, 4});

  check_get_data_subset(error_file, {0.4, 0.5, 0.55, 0.6}, {2, 2}, {1, 2}, 1,
                        2);

  check_get_data_subset(error_file, {0.4, 0.6, 0.55, 0.8, 0.66, 0.9}, {3, 2},
                        {1, 3}, 1, 3);

  check_get_data_subset(error_file, {0.0, 0.3, 0.11, 0.6}, {2, 2}, {0, 3}, 0,
                        2);

  check_get_data_subset(error_file, {}, {2, 0}, {}, 0, 2);

  check_get_data_subset(error_file, {}, {0, 2}, {0, 3}, 0, 0);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_dat_read() {
  const std::string h5_file_name("Unit.IO.H5.DatRead.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  std::vector<std::string> legend{"Time", "Error L2", "Error L1", "Error"};
  {
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    auto& error_file =
        my_file.insert<h5::Dat>("/L2_errors", legend, version_number);

    error_file.append([]() {
      Matrix result(3, 4);
      result(0, 0) = 0.0;
      result(0, 1) = 0.1;
      result(0, 2) = 0.2;
      result(0, 3) = 0.3;
      result(1, 0) = 0.11;
      result(1, 1) = 0.4;
      result(1, 2) = 0.5;
      result(1, 3) = 0.6;
      result(2, 0) = 0.22;
      result(2, 1) = 0.55;
      result(2, 2) = 0.6;
      result(2, 3) = 0.8;
      return result;
    }());
    error_file.append(Matrix(0, 0, 0.0));
    error_file.append(std::vector<double>{0.33, 0.66, 0.77, 0.90});
  }
  h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
  // No leading slash should also find the subfile, and a ".dat" extension as
  // well
  const auto& error_file = my_file.get<h5::Dat>("L2_errors.dat");
  CHECK(error_file.subfile_path() == "/L2_errors");

  // Check version info is correctly retrieved from Dat file
  CHECK(error_file.get_version() == version_number);

  // Check getting the header from Dat file
  std::stringstream ss;
  ss << "# ";
  auto build_info = info_from_build();
  ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
  const auto& header = error_file.get_header();
  CHECK("#\n# File created on "s ==
        header.substr(0, header.find("File created on ") + 16));
  CHECK(ss.str() == header.substr(header.find("# SpECTRE")));

  CHECK(error_file.get_legend() == legend);

  // Check data is retrieved correctly from Dat file
  std::array<hsize_t, 2> size_of_data{{4, 4}};
  CHECK(error_file.get_dimensions() == size_of_data);

  check_get_data(error_file,
                 {0.0, 0.1, 0.2, 0.3, 0.11, 0.4, 0.5, 0.6, 0.22, 0.55, 0.6, 0.8,
                  0.33, 0.66, 0.77, 0.9},
                 {4, 4});

  check_get_data_subset(error_file, {0.4, 0.5, 0.55, 0.6}, {2, 2}, {1, 2}, 1,
                        2);

  check_get_data_subset(error_file, {0.4, 0.6, 0.55, 0.8, 0.66, 0.9}, {3, 2},
                        {1, 3}, 1, 3);

  check_get_data_subset(error_file, {0.0, 0.3, 0.11, 0.6}, {2, 2}, {0, 3}, 0,
                        2);

  check_get_data_subset(error_file, {}, {2, 0}, {}, 0, 2);

  check_get_data_subset(error_file, {}, {0, 2}, {0, 3}, 0, 0);

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.IO.H5.Dat", "[Unit][IO][H5]") {
  test_errors();
  test_core_functionality();
  test_dat_read();
}
