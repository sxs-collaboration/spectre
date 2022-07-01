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
#include <typeinfo>
#include <utility>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TMPL.hpp"

// Test that we can read scalar, rank-1, rank-2, and rank-3 datasets
namespace {
void test_types_equal() {
  CHECK(h5::types_equal(h5::h5_type<double>(), h5::h5_type<double>()));
  CHECK_FALSE(h5::types_equal(h5::h5_type<double>(), h5::h5_type<float>()));
  CHECK_FALSE(h5::types_equal(h5::h5_type<char>(), h5::h5_type<float>()));
  CHECK_FALSE(h5::types_equal(h5::h5_type<int>(), h5::h5_type<float>()));
}

void test_read_data() {
  const std::string h5_file_name("Unit.IO.H5.ReadData.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const hid_t file_id = H5Fcreate(h5_file_name.c_str(), h5::h5f_acc_trunc(),
                                  h5::h5p_default(), h5::h5p_default());
  h5::detail::OpenGroup my_group(file_id, "ReadWrite",
                                 h5::AccessType::ReadWrite);
  const hid_t group_id = my_group.id();

  // Test writing rank1 datasets
  {
    const DataVector rank1_data{1.0, 2.0, 3.0};
    const std::string dataset_name{"rank1_dataset_datavector"};
    h5::write_data(group_id, rank1_data, dataset_name);
    const auto rank1_data_from_file =
        h5::read_data<1, DataVector>(group_id, dataset_name);
    CHECK(rank1_data_from_file == rank1_data);
  }
  const auto test_rank1_data = [&group_id](const auto& rank1_data,
                                           const std::vector<size_t>& extents,
                                           const std::string& dataset_name) {
    h5::write_data(group_id, rank1_data, extents, dataset_name);
    const auto rank1_data_from_file =
        h5::read_data<1, std::decay_t<decltype(rank1_data)>>(group_id,
                                                             dataset_name);
    CHECK(rank1_data_from_file == rank1_data);
  };
  test_rank1_data(std::vector<double>{1.0, 2.0, 3.0}, std::vector<size_t>{3},
                  "rank1_dataset_vector_double");
  test_rank1_data(std::vector<float>{1.0, 2.0, 3.0}, std::vector<size_t>{3},
                  "rank1_dataset_vector_float");

  // Test writing and reading `std::vector`s from rank-0 to rank-3
  using all_type_list =
      tmpl::list<double, int, unsigned int, long, unsigned long, long long,
                 unsigned long long>;

  h5::write_data(group_id, std::vector<double>{1.0 / 3.0}, {},
                 "scalar_dataset");
  CHECK(h5::read_data<0, double>(group_id, "scalar_dataset") == 1.0 / 3.0);
  h5::write_data(group_id, std::vector<float>{1.0f / 3.0f}, {},
                 "scalar_dataset_float");
  CHECK(h5::read_data<0, float>(group_id, "scalar_dataset_float") ==
        static_cast<float>(1.0 / 3.0));

  tmpl::for_each<all_type_list>([&group_id](auto x) {
    using DataType = typename decltype(x)::type;
    const std::vector<DataType> rank1_data{1, 2, 3};
    const std::vector<size_t> rank1_extents{3};
    h5::write_data<DataType>(
        group_id, rank1_data, rank1_extents,
        "rank1_dataset_" + std::string(typeid(DataType).name()));

    std::vector<DataType> rank1_data_from_file =
        h5::read_data<1, std::vector<DataType>>(
            group_id, "rank1_dataset_" + std::string(typeid(DataType).name()));
    CHECK(rank1_data_from_file == std::vector<DataType>{1, 2, 3});
  });

  tmpl::for_each<all_type_list>([&group_id](auto x) {
    using DataType = typename decltype(x)::type;
    const std::vector<DataType> rank2_data{1, 2, 3, 4};
    const std::vector<size_t> rank2_extents{2, 2};
    h5::write_data<DataType>(
        group_id, rank2_data, rank2_extents,
        "rank2_dataset_" + std::string(typeid(DataType).name()));

    boost::multi_array<DataType, 2> rank2_data_from_file =
        h5::read_data<2, boost::multi_array<DataType, 2>>(
            group_id, "rank2_dataset_" + std::string(typeid(DataType).name()));
    boost::multi_array<DataType, 2> expected_rank2_data(boost::extents[2][2]);
    expected_rank2_data[0][0] = static_cast<DataType>(1);
    expected_rank2_data[0][1] = static_cast<DataType>(2);
    expected_rank2_data[1][0] = static_cast<DataType>(3);
    expected_rank2_data[1][1] = static_cast<DataType>(4);
    CHECK(rank2_data_from_file == expected_rank2_data);
  });

  tmpl::for_each<all_type_list>([&group_id](auto x) {
    using DataType = typename decltype(x)::type;
    const std::vector<DataType> rank3_data{1, 2, 3, 4, 5, 6};
    const std::vector<size_t> rank3_extents{1, 2, 3};
    h5::write_data<DataType>(
        group_id, rank3_data, rank3_extents,
        "rank3_dataset_" + std::string(typeid(DataType).name()));

    boost::multi_array<DataType, 3> rank3_data_from_file =
        h5::read_data<3, boost::multi_array<DataType, 3>>(
            group_id, "rank3_dataset_" + std::string(typeid(DataType).name()));
    boost::multi_array<DataType, 3> expected_rank3_data(
        boost::extents[1][2][3]);
    expected_rank3_data[0][0][0] = static_cast<DataType>(1);
    expected_rank3_data[0][0][1] = static_cast<DataType>(2);
    expected_rank3_data[0][0][2] = static_cast<DataType>(3);
    expected_rank3_data[0][1][0] = static_cast<DataType>(4);
    expected_rank3_data[0][1][1] = static_cast<DataType>(5);
    expected_rank3_data[0][1][2] = static_cast<DataType>(6);
    CHECK(rank3_data_from_file == expected_rank3_data);
  });

  CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << h5_file_name << "'");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

// Check that we can insert and open subfiles at the '/' level
void test_check_if_object_exists() {
  const std::string h5_file_name("Unit.IO.H5.check_if_object_exists.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  {
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    my_file.insert<h5::Header>("/");
  }

  // Reopen the file to check that the subfile '/' can be opened
  h5::H5File<h5::AccessType::ReadWrite> reopened_file(h5_file_name, true);
  reopened_file.get<h5::Header>("/");
  CHECK(file_system::check_if_file_exists(h5_file_name) == true);
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_contains_attribute_false() {
  const std::string h5_file_name("Unit.IO.H5.contains_attribute_false.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  const hid_t file_id = H5Fcreate(h5_file_name.c_str(), h5::h5f_acc_trunc(),
                                  h5::h5p_default(), h5::h5p_default());
  CHECK_H5(file_id, "Failed to open file: " << h5_file_name);
  CHECK_FALSE(h5::contains_attribute(file_id, "/", "no_attr"));
  CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << h5_file_name << "'");
}

void test_errors() {
  CHECK_THROWS_WITH(
      []() {
        const std::string file_name("Unit.IO.H5.read_data_error.h5");
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        const hid_t file_id = H5Fcreate(file_name.c_str(), h5::h5f_acc_trunc(),
                                        h5::h5p_default(), h5::h5p_default());
        CHECK_H5(file_id, "Failed to open file: " << file_name);
        static_cast<void>(h5::read_data<1, DataVector>(file_id, "no_dataset"));
        CHECK_H5(H5Fclose(file_id),
                 "Failed to close file: '" << file_name << "'");
      }(),
      Catch::Contains(
          "Failed HDF5 operation: Failed to open dataset 'no_dataset'"));
  if (file_system::check_if_file_exists("Unit.IO.H5.read_data_error.h5")) {
    file_system::rm("Unit.IO.H5.read_data_error.h5", true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5", "[Unit][IO][H5]") {
  test_types_equal();
  test_read_data();
  test_check_if_object_exists();
  test_contains_attribute_false();
  test_errors();
}
