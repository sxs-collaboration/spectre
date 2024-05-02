// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Cce.hpp"
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
void test_errors() {
  std::string file_name{"./Unit.IO.H5.FileErrorObjectAlreadyExists.h5"};
  CHECK_THROWS_WITH(
      ([&file_name]() {
        const uint32_t version_number = 4;
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file =
              my_file.insert<h5::Cce>("/Bondi", l_max, version_number);
          (void)cce_file;
          my_file.close_current_object();
        }
        { my_file.insert<h5::Cce>("/Bondi//", l_max, version_number); }
      }()),
      Catch::Matchers::ContainsSubstring(
          "Cannot insert an Object that already exists. Failed to "
          "add Object named: /Bondi"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.FileErrorObjectAlreadyOpenGet.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& cce_file = my_file.insert<h5::Cce>("/DummyPath", l_max);
        auto& get_cce_file = my_file.get<h5::Cce>("/DummyPath", l_max);
        (void)cce_file;
        (void)get_cce_file;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot open object /DummyPath."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.FileErrorObjectAlreadyOpenInsert.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& cce_file = my_file.insert<h5::Cce>("/DummyPath", l_max);
        auto& cce_file2 = my_file.insert<h5::Cce>("/DummyPath2", l_max);
        (void)cce_file;
        (void)cce_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot insert object "
          "/DummyPath2."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.FileErrorObjectAlreadyOpenTryInsert.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        auto& cce_file = my_file.insert<h5::Cce>("/DummyPath", l_max);
        auto& cce_file2 = my_file.try_insert<h5::Cce>("/DummyPath2", l_max);
        (void)cce_file;
        (void)cce_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /DummyPath already open. Cannot try to insert "
          "object /DummyPath2."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.FileErrorObjectAlreadyOpen.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/DummyPath", l_max);
          my_file.close_current_object();
          (void)cce_file;
          auto& cce_file2 = my_file.insert<h5::Cce>("/Dummy/Path2", l_max);
          my_file.close_current_object();
          (void)cce_file2;
        }
        auto& cce_file2 = my_file.get<h5::Cce>("/Dummy/Path2", l_max);
        auto& cce_file = my_file.get<h5::Cce>("/DummyPath", l_max);
        (void)cce_file;
        (void)cce_file2;
      }(),
      Catch::Matchers::ContainsSubstring(
          "Object /Dummy/Path2 already open. Cannot open object /DummyPath."));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.LegendsDontMatch.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        const size_t l_max = 4;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/DummyPath", l_max);
          my_file.close_current_object();
          (void)cce_file;
          auto& cce_file2 = my_file.get<h5::Cce>("/DummyPath", l_max + 1);
          (void)cce_file2;
        }
      }(),
      Catch::Matchers::ContainsSubstring("l_max from cce file") and
          Catch::Matchers::ContainsSubstring(
              "does not match l_max in constructor"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.NoBondiVariable.h5";
  CHECK_THROWS_WITH(
      ([&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        // Easy to make data
        const size_t l_max = 0;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/Bondi", l_max);
          std::unordered_map<std::string, std::vector<double>> data{};
          data["News"] = std::vector<double>(3, 0.0);
          cce_file.append(data);
        }
      }()),
      Catch::Matchers::ContainsSubstring(
          "Passed in data does not contain the bondi variable"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.IncorrectDataSize.h5";
  CHECK_THROWS_WITH(
      ([&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        // Easy to make data
        const size_t l_max = 0;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/Bondi", l_max);
          std::unordered_map<std::string, std::vector<double>> data{};
          // Incorrect vector length. Need all the vars or else we'll hit the
          // previous error
          data["EthInertialRetardedTime"] = std::vector<double>(10, 0.0);
          data["News"] = std::vector<double>(10, 0.0);
          data["Psi0"] = std::vector<double>(10, 0.0);
          data["Psi1"] = std::vector<double>(10, 0.0);
          data["Psi2"] = std::vector<double>(10, 0.0);
          data["Psi3"] = std::vector<double>(10, 0.0);
          data["Psi4"] = std::vector<double>(10, 0.0);
          data["Strain"] = std::vector<double>(10, 0.0);
          cce_file.append(data);
        }
      }()),
      Catch::Matchers::ContainsSubstring(
          "Cannot add columns to Cce files. Current number of columns is 3 but "
          "received 10 entries"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.LTooLarge.h5";
  CHECK_THROWS_WITH(
      ([&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        // Easy to make data
        const size_t l_max = 0;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/Bondi", l_max);
          std::unordered_map<std::string, std::vector<double>> data{};
          data["EthInertialRetardedTime"] = std::vector<double>(3, 0.0);
          data["News"] = std::vector<double>(3, 0.0);
          data["Psi0"] = std::vector<double>(3, 0.0);
          data["Psi1"] = std::vector<double>(3, 0.0);
          data["Psi2"] = std::vector<double>(3, 0.0);
          data["Psi3"] = std::vector<double>(3, 0.0);
          data["Psi4"] = std::vector<double>(3, 0.0);
          data["Strain"] = std::vector<double>(3, 0.0);
          cce_file.append(data);
          const auto subset = cce_file.get_data_subset("Psi0", {2}, 0);
          (void)subset;
        }
      }()),
      Catch::Matchers::ContainsSubstring("One (or more) of the requested ells "
                                         "(2) is larger than the l_max 0"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  file_name = "./Unit.IO.H5.IncorrectBondiVarDim.h5";
  CHECK_THROWS_WITH(
      [&file_name]() {
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        // Easy to make data
        const size_t l_max = 0;
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        {
          auto& cce_file = my_file.insert<h5::Cce>("/Bondi", l_max);
          const auto size = cce_file.get_dimensions("NonExistentBondiVar");
          (void)size;
        }
      }(),
      Catch::Matchers::ContainsSubstring(
          "Requested bondi variable NonExistentBondiVar not available"));
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
}

void check_written_data(
    const h5::Cce& cce_file,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>
        expected_data,
    const std::unordered_set<std::string>& bondi_variables,
    const std::vector<std::string>& legend, const size_t version_number) {
  // Check version info is correctly retrieved from Cce file
  CHECK(cce_file.get_version() == version_number);

  // Check getting the header from Cce file
  std::stringstream ss;
  ss << "# ";
  auto build_info = info_from_build();
  ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
  const auto& header = cce_file.get_header();
  CHECK(header.starts_with("#\n# File created on "));
  CHECK(header.ends_with(ss.str()));

  CHECK(cce_file.get_legend() == legend);

  // Check data is retrieved correctly from Cce file
  std::array<hsize_t, 2> size_of_data{{4, legend.size()}};
  for (const std::string& bondi_var : bondi_variables) {
    CHECK(cce_file.get_dimensions(bondi_var) == size_of_data);
  }

  const std::unordered_map<std::string, Matrix> data_in_cce_file =
      [&expected_data, &legend, &bondi_variables]() {
        std::unordered_map<std::string, Matrix> result{};
        for (const std::string& bondi_var : bondi_variables) {
          Matrix matrix_result(4, legend.size());
          for (size_t row = 0; row < matrix_result.rows(); row++) {
            for (size_t col = 0; col < legend.size(); col++) {
              matrix_result(row, col) = expected_data.at(bondi_var)[row][col];
            }
          }

          result[bondi_var] = std::move(matrix_result);
        }
        return result;
      }();

  CHECK(cce_file.get_data() == data_in_cce_file);
  for (const std::string& bondi_var : bondi_variables) {
    CHECK(cce_file.get_data(bondi_var) == data_in_cce_file.at(bondi_var));
  }

  // Just ell = 2
  {
    const std::unordered_map<std::string, Matrix> expected_subset =
        [&expected_data, &bondi_variables]() {
          std::unordered_map<std::string, Matrix> result{};
          for (const std::string& bondi_var : bondi_variables) {
            Matrix matrix_result(2, 11);
            size_t row = 1;
            for (size_t row_idx = 0; row_idx < 2; row_idx++, row++) {
              // time
              matrix_result(row_idx, 0) = expected_data.at(bondi_var)[row][0];
              size_t col = 9;
              for (size_t col_idx = 1; col_idx < 11; col_idx++, col++) {
                matrix_result(row_idx, col_idx) =
                    expected_data.at(bondi_var)[row][col];
              }
            }

            result[bondi_var] = std::move(matrix_result);
          }

          return result;
        }();

    const auto subset = cce_file.get_data_subset({2}, 1, 2);
    CHECK(subset == expected_subset);
    for (const std::string& bondi_var : bondi_variables) {
      CHECK(cce_file.get_data_subset(bondi_var, {2}, 1, 2) ==
            expected_subset.at(bondi_var));
    }
  }
  // ell = 1,3
  {
    const std::unordered_map<std::string, Matrix> expected_subset =
        [&expected_data, &bondi_variables]() {
          std::unordered_map<std::string, Matrix> result{};
          for (const std::string& bondi_var : bondi_variables) {
            Matrix matrix_result(3, 21);
            size_t row = 1;
            for (size_t row_idx = 0; row_idx < 3; row_idx++, row++) {
              // time
              matrix_result(row_idx, 0) = expected_data.at(bondi_var)[row][0];
              // ell = 1
              size_t col = 3;
              for (size_t col_idx = 1; col_idx < 7; col_idx++, col++) {
                matrix_result(row_idx, col_idx) =
                    expected_data.at(bondi_var)[row][col];
              }
              // ell = 3
              col = 19;
              for (size_t col_idx = 7; col_idx < 21; col_idx++, col++) {
                matrix_result(row_idx, col_idx) =
                    expected_data.at(bondi_var)[row][col];
              }
            }

            result[bondi_var] = std::move(matrix_result);
          }

          return result;
        }();

    const auto subset = cce_file.get_data_subset({1, 3}, 1, 3);
    CHECK(subset == expected_subset);
    for (const std::string& bondi_var : bondi_variables) {
      CHECK(cce_file.get_data_subset(bondi_var, {1, 3}, 1, 3) ==
            expected_subset.at(bondi_var));
    }
  }
  // ell = 0,2
  {
    const std::unordered_map<std::string, Matrix> expected_subset =
        [&expected_data, &bondi_variables]() {
          std::unordered_map<std::string, Matrix> result{};
          for (const std::string& bondi_var : bondi_variables) {
            Matrix matrix_result(2, 13);
            size_t row = 0;
            for (size_t row_idx = 0; row_idx < 2; row_idx++, row++) {
              // time
              matrix_result(row_idx, 0) = expected_data.at(bondi_var)[row][0];
              // ell = 0
              size_t col = 1;
              for (size_t col_idx = 1; col_idx < 3; col_idx++, col++) {
                matrix_result(row_idx, col_idx) =
                    expected_data.at(bondi_var)[row][col];
              }

              // ell = 2
              col = 9;
              for (size_t col_idx = 3; col_idx < 13; col_idx++, col++) {
                matrix_result(row_idx, col_idx) =
                    expected_data.at(bondi_var)[row][col];
              }
            }

            result[bondi_var] = std::move(matrix_result);
          }

          return result;
        }();

    const auto subset = cce_file.get_data_subset({0, 2}, 0, 2);
    CHECK(subset == expected_subset);
    for (const std::string& bondi_var : bondi_variables) {
      CHECK(cce_file.get_data_subset(bondi_var, {0, 2}, 0, 2) ==
            expected_subset.at(bondi_var));
    }
  }
  // No ell
  {
    const auto subset = cce_file.get_data_subset({}, 0, 2);
    const Matrix answer(2, 0, 0.0);
    const std::unordered_map<std::string, Matrix> expected_subset =
        [&answer, &bondi_variables]() {
          std::unordered_map<std::string, Matrix> result{};
          for (const std::string& bondi_var : bondi_variables) {
            result[bondi_var] = answer;
          }
          return result;
        }();
    CHECK(subset == expected_subset);
    for (const std::string& bondi_var : bondi_variables) {
      CHECK(cce_file.get_data_subset(bondi_var, {}, 0, 2) == answer);
    }
  }
  // No rows
  {
    const auto subset = cce_file.get_data_subset({0, 3}, 0, 0);
    const Matrix answer(0, 17, 0.0);
    const std::unordered_map<std::string, Matrix> expected_subset =
        [&answer, &bondi_variables]() {
          std::unordered_map<std::string, Matrix> result{};
          for (const std::string& bondi_var : bondi_variables) {
            result[bondi_var] = answer;
          }
          return result;
        }();
    CHECK(subset == expected_subset);
    for (const std::string& bondi_var : bondi_variables) {
      CHECK(cce_file.get_data_subset(bondi_var, {0, 3}, 0, 0) == answer);
    }
  }
}

template <typename Generator>
void test_core_functionality(const gsl::not_null<Generator*> generator) {
  const std::unordered_set<std::string> bondi_variables{
      "EthInertialRetardedTime",
      "News",
      "Psi0",
      "Psi1",
      "Psi2",
      "Psi3",
      "Psi4",
      "Strain"};

  const std::string h5_file_name("Unit.IO.H5.Cce.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const size_t l_max = 4;
  std::vector<std::string> legend{
      "time",        "Real Y_0,0",  "Imag Y_0,0",  "Real Y_1,-1", "Imag Y_1,-1",
      "Real Y_1,0",  "Imag Y_1,0",  "Real Y_1,1",  "Imag Y_1,1",  "Real Y_2,-2",
      "Imag Y_2,-2", "Real Y_2,-1", "Imag Y_2,-1", "Real Y_2,0",  "Imag Y_2,0",
      "Real Y_2,1",  "Imag Y_2,1",  "Real Y_2,2",  "Imag Y_2,2",  "Real Y_3,-3",
      "Imag Y_3,-3", "Real Y_3,-2", "Imag Y_3,-2", "Real Y_3,-1", "Imag Y_3,-1",
      "Real Y_3,0",  "Imag Y_3,0",  "Real Y_3,1",  "Imag Y_3,1",  "Real Y_3,2",
      "Imag Y_3,2",  "Real Y_3,3",  "Imag Y_3,3",  "Real Y_4,-4", "Imag Y_4,-4",
      "Real Y_4,-3", "Imag Y_4,-3", "Real Y_4,-2", "Imag Y_4,-2", "Real Y_4,-1",
      "Imag Y_4,-1", "Real Y_4,0",  "Imag Y_4,0",  "Real Y_4,1",  "Imag Y_4,1",
      "Real Y_4,2",  "Imag Y_4,2",  "Real Y_4,3",  "Imag Y_4,3",  "Real Y_4,4",
      "Imag Y_4,4"};

  const auto create_data =
      [&legend, &generator](const size_t row) -> std::vector<double> {
    std::uniform_real_distribution<double> dist{static_cast<double>(row),
                                                static_cast<double>(row + 1)};
    std::vector<double> result(legend.size());
    result[0] = static_cast<double>(row);
    for (size_t i = 1; i < legend.size(); i++) {
      result[i] = dist(*generator);
    }
    return result;
  };

  std::unordered_map<std::string, std::vector<std::vector<double>>>
      expected_data{};
  {
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    my_file.insert<h5::Cce>("/Bondi", l_max, version_number);
    my_file.close_current_object();

    // Check that the Cce file is found to be a subgroup of the file.
    const auto groups_in_file = my_file.groups();
    CHECK(alg::find(groups_in_file, std::string{"Bondi.cce"}) !=
          groups_in_file.end());

    auto& cce_file = my_file.get<h5::Cce>("/Bondi", l_max, version_number);

    CHECK(legend == cce_file.get_legend());

    std::unordered_map<std::string, std::vector<double>> tmp_data{};
    for (const std::string& bondi_var : bondi_variables) {
      expected_data[bondi_var] = std::vector<std::vector<double>>(4);
      tmp_data[bondi_var];
    }

    for (size_t i = 0; i < 4; i++) {
      for (const std::string& bondi_var : bondi_variables) {
        tmp_data.at(bondi_var) = create_data(i);
        expected_data.at(bondi_var)[i] = tmp_data.at(bondi_var);
      }
      cce_file.append(tmp_data);
    }

    // Test with ReadWrite access
    check_written_data(cce_file, expected_data, bondi_variables, legend,
                       version_number);
  }

  // Test with ReadOnly access
  {
    h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
    const auto& cce_file =
        my_file.get<h5::Cce>("Bondi.cce", l_max, version_number);

    check_written_data(cce_file, expected_data, bondi_variables, legend,
                       version_number);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
}  // namespace

// [[TimeOut, 10]]
SPECTRE_TEST_CASE("Unit.IO.H5.Cce", "[Unit][IO][H5]") {
  MAKE_GENERATOR(generator);
  test_errors();
  test_core_functionality(make_not_null(&generator));
}
