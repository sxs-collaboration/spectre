// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "IO/Connectivity.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Version.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/GetOutput.hpp"

SPECTRE_TEST_CASE("Unit.IO.H5.File", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.File.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  /// [h5file_readwrite_get_header]
  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  // Check that the header was written correctly
  const std::string& header = my_file.get<h5::Header>("/header").get_header();
  /// [h5file_readwrite_get_header]

  CHECK(my_file.name() == h5_file_name);

  std::stringstream ss;
  ss << "# ";
  auto build_info = info_from_build();
  ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");

  CHECK("#\n# File created on "s ==
        header.substr(0, header.find("File created on ") + 16));
  CHECK(ss.str() == header.substr(header.find("# SpECTRE Build Information:")));

  my_file.close_current_object();

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.FileMove", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.FileMove.h5");
  const std::string h5_file_name2("Unit.IO.H5.FileMove2.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  if (file_system::check_if_file_exists(h5_file_name2)) {
    file_system::rm(h5_file_name2, true);
  }

  auto my_file =
      std::make_unique<h5::H5File<h5::AccessType::ReadWrite>>(h5_file_name);

  auto my_file2 = std::make_unique<h5::H5File<h5::AccessType::ReadWrite>>(
      std::move(*my_file));
  my_file.reset();
  CHECK(my_file == nullptr);

  h5::H5File<h5::AccessType::ReadWrite> my_file3(h5_file_name2);
  my_file3 = std::move(*my_file2);
  my_file2.reset();
}

// [[OutputRegex, Cannot open the object '/Dummy.hdr' because it does not
// exist.]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorObjectNotExist", "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string file_name = "./Unit.IO.H5.FileErrorObjectNotExist.h5";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
  my_file.get<h5::Header>("/Dummy").get_header();
}

// [[OutputRegex, All HDF5 file names must end in '.h5'. The path and file name
// './Unit.IO.H5.FileErrorNotH5.h5ab' does not satisfy this]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorNotH5", "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string file_name = "./Unit.IO.H5.FileErrorNotH5.h5ab";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
}

// [[OutputRegex, Cannot create a file in ReadOnly mode,
// './Unit.IO.H5.FileErrorFileNotExist.h5']]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorFileNotExist", "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string file_name = "./Unit.IO.H5.FileErrorFileNotExist.h5";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  h5::H5File<h5::AccessType::ReadOnly> my_file(file_name);
}

// [[OutputRegex, Cannot append to a file opened in read-only mode. File name
// is:
// ./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorCannotAppendReadOnly",
                  "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string file_name = "./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  { h5::H5File<h5::AccessType::ReadWrite> my_file(file_name); }
  h5::H5File<h5::AccessType::ReadOnly> my_file(file_name, true);
}

/// [willfail_example_for_dev_doc]
// [[OutputRegex, File './Unit.IO.H5.FileErrorExists.h5' already exists and we
// are not allowed to append. To reduce the risk of accidental deletion you must
// explicitly delete the file first using the file_system library in
// SpECTRE or through your shell.]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorExists", "[Unit][IO][H5]") {
  ERROR_TEST();
  /// [willfail_example_for_dev_doc]
  std::string file_name = "./Unit.IO.H5.FileErrorExists.h5";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  // Need to close file opened by my_file otherwise we get the error
  // pure virtual method called
  // terminate called recursively
  {
    /// [h5file_readwrite_file]
    h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
    /// [h5file_readwrite_file]
  }
  h5::H5File<h5::AccessType::ReadWrite> my_file_2(file_name);
}

// [[OutputRegex, Cannot open the object '/Dummy.hdr' because it does not
// exist.]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorObjectNotExistConst", "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string file_name = "./Unit.IO.H5.FileErrorObjectNotExistConst.h5";
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }
  const h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
  my_file.get<h5::Header>("/Dummy").get_header();
}

// [[OutputRegex, Cannot insert an Object that already exists. Failed to add
// Object named: /L2_errors]]
SPECTRE_TEST_CASE("Unit.IO.H5.FileErrorObjectAlreadyExists", "[Unit][IO][H5]") {
  ERROR_TEST();
  std::string h5_file_name = "./Unit.IO.H5.FileErrorObjectAlreadyExists.h5";
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  std::vector<std::string> legend{"Time", "Error L2", "Error L1", "Error"};
  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  {
    auto& error_file =
        my_file.insert<h5::Dat>("/L2_errors///", legend, version_number);
    error_file.append(std::vector<double>{0, 0.1, 0.2, 0.3});
  }
  {
    my_file.insert<h5::Dat>("/L2_errors//", legend, version_number);
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.Version", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.Version.h5");
  const uint32_t version_number = 2;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  {
    /// [h5file_write_version]
    h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
    my_file.insert<h5::Version>("/the_version", version_number);
    /// [h5file_write_version]
  }

  h5::H5File<h5::AccessType::ReadOnly> my_file(h5_file_name);
  /// [h5file_read_version]
  const auto& const_version = my_file.get<h5::Version>("/the_version");
  /// [h5file_read_version]
  CHECK(version_number == const_version.get_version());
  auto& version = my_file.get<h5::Version>("/the_version");
  CHECK(version_number == version.get_version());
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.Dat", "[Unit][IO][H5]") {
  const std::string h5_file_name("Unit.IO.H5.Dat.h5");
  const uint32_t version_number = 4;
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  std::vector<std::string> legend{"Time", "Error L2", "Error L1", "Error"};

  h5::H5File<h5::AccessType::ReadWrite> my_file(h5_file_name);
  my_file.insert<h5::Dat>("/L2_errors", legend, version_number);
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

  /// [h5dat_get_data]
  const Matrix data_in_dat_file = []() {
    Matrix result(4, 4);
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
    result(3, 0) = 0.33;
    result(3, 1) = 0.66;
    result(3, 2) = 0.77;
    result(3, 3) = 0.9;
    return result;
  }();
  CHECK(error_file.get_data() == data_in_dat_file);
  /// [h5dat_get_data]

  {
    const auto subset = error_file.get_data_subset({1, 2}, 1, 2);
    const Matrix answer = []() {
      Matrix result(2, 2);
      result(0, 0) = 0.4;
      result(0, 1) = 0.5;
      result(1, 0) = 0.55;
      result(1, 1) = 0.6;
      return result;
    }();
    CHECK(subset == answer);
  }
  {
    /// [h5dat_get_subset]
    const auto subset = error_file.get_data_subset({1, 3}, 1, 3);
    const Matrix answer = []() {
      Matrix result(3, 2);
      result(0, 0) = 0.4;
      result(0, 1) = 0.6;
      result(1, 0) = 0.55;
      result(1, 1) = 0.8;
      result(2, 0) = 0.66;
      result(2, 1) = 0.9;
      return result;
    }();
    CHECK(subset == answer);
    /// [h5dat_get_subset]
  }
  {
    const auto subset = error_file.get_data_subset({0, 3}, 0, 2);
    const Matrix answer = []() {
      Matrix result(2, 2);
      result(0, 0) = 0.0;
      result(0, 1) = 0.3;
      result(1, 0) = 0.11;
      result(1, 1) = 0.6;
      return result;
    }();
    CHECK(subset == answer);
  }
  {
    const auto subset = error_file.get_data_subset({}, 0, 2);
    const Matrix answer(2, 0, 0.0);
    CHECK(subset == answer);
  }
  {
    const auto subset = error_file.get_data_subset({0, 3}, 0, 0);
    const Matrix answer(0, 2, 0.0);
    CHECK(subset == answer);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.DatRead", "[Unit][IO][H5]") {
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
  const auto& error_file = my_file.get<h5::Dat>("/L2_errors");

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

  const Matrix data_in_dat_file = []() {
    Matrix result(4, 4);
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
    result(3, 0) = 0.33;
    result(3, 1) = 0.66;
    result(3, 2) = 0.77;
    result(3, 3) = 0.9;
    return result;
  }();
  CHECK(error_file.get_data() == data_in_dat_file);

  {
    const auto subset = error_file.get_data_subset({1, 2}, 1, 2);
    const Matrix answer = []() {
      Matrix result(2, 2);
      result(0, 0) = 0.4;
      result(0, 1) = 0.5;
      result(1, 0) = 0.55;
      result(1, 1) = 0.6;
      return result;
    }();
    CHECK(subset == answer);
  }
  {
    const auto subset = error_file.get_data_subset({1, 3}, 1, 3);
    const Matrix answer = []() {
      Matrix result(3, 2);
      result(0, 0) = 0.4;
      result(0, 1) = 0.6;
      result(1, 0) = 0.55;
      result(1, 1) = 0.8;
      result(2, 0) = 0.66;
      result(2, 1) = 0.9;
      return result;
    }();
    CHECK(subset == answer);
  }
  {
    const auto subset = error_file.get_data_subset({0, 3}, 0, 2);
    const Matrix answer = []() {
      Matrix result(2, 2);
      result(0, 0) = 0.0;
      result(0, 1) = 0.3;
      result(1, 0) = 0.11;
      result(1, 1) = 0.6;
      return result;
    }();
    CHECK(subset == answer);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.contains_attribute_false", "[Unit][IO][H5]") {
  const std::string file_name("Unit.IO.H5.contains_attribute_false.h5");
  const hid_t file_id =
      H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_H5(file_id, "Failed to open file: " << file_name);
  CHECK_FALSE(h5::contains_attribute(file_id, "/", "no_attr"));
  CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
}

/// [[OutputRegex, could not open dataset 'no_dataset']]
SPECTRE_TEST_CASE("Unit.IO.H5.read_data_error", "[Unit][IO][H5]") {
  ERROR_TEST();
  const std::string file_name("Unit.IO.H5.read_data_error.h5");
  const hid_t file_id =
      H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_H5(file_id, "Failed to open file: " << file_name);
  static_cast<void>(h5::read_data(file_id, "no_dataset"));
  CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
}

// [[OutputRegex, Cannot create group 'group' in path: /group because the access
// is ReadOnly]]
SPECTRE_TEST_CASE("Unit.IO.H5.OpenGroup_read_access", "[Unit][IO][H5]") {
  ERROR_TEST();
  const std::string file_name("Unit.IO.H5.OpenGroup_read_access.h5");

  const hid_t file_id =
      H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  CHECK_H5(file_id, "Failed to open file: " << file_name);
  { h5::detail::OpenGroup group(file_id, "/group", h5::AccessType::ReadOnly); }
  CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
}

// [[OutputRegex, Failed to open the group 'group' because the file_id passed in
// is invalid, or because the group_id inside the OpenGroup constructor got
// corrupted. It is most likely that the file_id is invalid.]]
SPECTRE_TEST_CASE("Unit.IO.H5.OpenGroup_bad_group_id", "[Unit][IO][H5]") {
  ERROR_TEST();
  h5::detail::OpenGroup group(-1, "/group", h5::AccessType::ReadWrite);
}

SPECTRE_TEST_CASE("Unit.IO.H5.OpenGroupMove", "[Unit][IO][H5]") {
  const std::string file_name("Unit.IO.H5.OpenGroupMove.h5");
  const std::string file_name2("Unit.IO.H5.OpenGroupMove2.h5");

  {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    const hid_t file_id =
        H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_H5(file_id, "Failed to open file: " << file_name);
    auto group = std::make_unique<h5::detail::OpenGroup>(
        file_id, "/group/group2/group3", h5::AccessType::ReadWrite);
    h5::detail::OpenGroup group2(std::move(*group));
    group.reset();
    CHECK(group == nullptr);
    CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
  }

  {
    if (file_system::check_if_file_exists(file_name)) {
      file_system::rm(file_name, true);
    }
    if (file_system::check_if_file_exists(file_name2)) {
      file_system::rm(file_name2, true);
    }
    const hid_t file_id =
        H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_H5(file_id, "Failed to open file: " << file_name);
    const hid_t file_id2 =
        H5Fcreate(file_name2.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    CHECK_H5(file_id2, "Failed to open file: " << file_name);
    {
      h5::detail::OpenGroup group(file_id, "/group/group2/group3",
                                  h5::AccessType::ReadWrite);
      h5::detail::OpenGroup group2(file_id2, "/group/group2/group3",
                                   h5::AccessType::ReadWrite);
      group2 = std::move(group);
      h5::detail::OpenGroup group3(file_id2, "/group/group2/group3",
                                   h5::AccessType::ReadWrite);
    }
    CHECK_H5(H5Fclose(file_id), "Failed to close file: '" << file_name << "'");
    CHECK_H5(H5Fclose(file_id2),
             "Failed to close file: '" << file_name2 << "'");
  }
}

SPECTRE_TEST_CASE("Unit.IO.H5.TopologyStreams", "[Unit][IO][H5]") {
  CHECK(get_output(vis::detail::Topology::Line) == "Line"s);
  CHECK(get_output(vis::detail::Topology::Quad) == "Quad"s);
  CHECK(get_output(vis::detail::Topology::Hexahedron) == "Hexahedron"s);
}
