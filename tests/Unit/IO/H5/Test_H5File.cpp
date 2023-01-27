// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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

#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/SourceArchive.hpp"
#include "IO/H5/Version.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/MakeString.hpp"

namespace {
void test_access_type() {
  CHECK(get_output(h5::AccessType::ReadOnly) == "ReadOnly");
  CHECK(get_output(h5::AccessType::ReadWrite) == "ReadWrite");
}

void test_core_functionality() {
  const std::string h5_file_name("Unit.IO.H5.File.h5");
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  // [h5file_readwrite_get_header]
  std::string input_source_yaml{"TestOption: 4\n\nOtherOption: 5\n\n"};
  // Add a giant set of comments to the h5 file, to test that we can
  // read and write very long yaml files to H5 attributes
  input_source_yaml += "# "s;
  for (size_t i = 0; i < 1000000; ++i) {
    input_source_yaml += "abcdefghij"s;
    // Output lines < 80 chars, as in an actual yaml input file
    if (i % 7 == 0) {
      input_source_yaml += "\n# "s;
    }
  }
  input_source_yaml += "\n\n"s;

  h5::H5File<h5::AccessType::ReadWrite> my_file0(h5_file_name, false,
                                                 input_source_yaml);
  // Check that the header was written correctly
  const std::string header = my_file0.get<h5::Header>("/header").get_header();
  // [h5file_readwrite_get_header]
  my_file0.close_current_object();

  const auto check_header = [&h5_file_name,
                             &input_source_yaml](const auto& my_file) {
    const auto& header_obj = my_file.template get<h5::Header>("/header");
    CHECK(header_obj.subfile_path() == "/header");
    // Check that the header was written correctly
    const std::string& my_header = header_obj.get_header();

    CHECK(my_file.name() == h5_file_name);

    // Check that the subgroups were found correctly
    std::vector<std::string> h5_file_groups{"src.tar.gz"};
    CHECK(my_file.groups() == h5_file_groups);

    CHECK("#\n# File created on "s ==
          my_header.substr(0, my_header.find("File created on ") + 16));
    CHECK(my_header.substr(my_header.find("# SpECTRE Build Information:")) ==
          std::string{MakeString{}
                      << "# "
                      << std::regex_replace(info_from_build(), std::regex{"\n"},
                                            "\n# ")});
    CHECK(header_obj.get_env_variables() ==
          formaline::get_environment_variables());
    CHECK(header_obj.get_build_info() == formaline::get_build_info());

    CHECK(my_file.input_source() == input_source_yaml);
  };
  check_header(my_file0);
  my_file0.close_current_object();
  check_header(h5::H5File<h5::AccessType::ReadOnly>(h5_file_name));
  my_file0.close_current_object();

  const auto check_source_archive = [](const auto& my_file) {
    const std::vector<char> archive =
        my_file.template get<h5::SourceArchive>("/src").get_archive();
    CHECK(archive == formaline::get_archive());
  };
  check_source_archive(my_file0);
  my_file0.close_current_object();
  check_source_archive(h5::H5File<h5::AccessType::ReadOnly>(h5_file_name));
  my_file0.close_current_object();

  my_file0.close();

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}

void test_file_move() {
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

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  if (file_system::check_if_file_exists(h5_file_name2)) {
    file_system::rm(h5_file_name2, true);
  }
}

void test_error_messages() {
  CHECK_THROWS_WITH(
      []() {
        const std::string file_name = "./Unit.IO.H5.FileErrorObjectNotExist.h5";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        };
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        my_file.get<h5::Header>("/Dummy").get_header();
      }(),
      Catch::Contains(
          "Cannot open the object '/Dummy.hdr' because it does not exist."));
  if (file_system::check_if_file_exists(
          "./Unit.IO.H5.FileErrorObjectNotExist.h5")) {
    file_system::rm("./Unit.IO.H5.FileErrorObjectNotExist.h5", true);
  }

  CHECK_THROWS_WITH(
      []() {
        const std::string file_name = "./Unit.IO.H5.FileErrorNotH5.h5ab";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
      }(),
      Catch::Contains(
          "All HDF5 file names must end in '.h5'. The path and file name "
          "'./Unit.IO.H5.FileErrorNotH5.h5ab' does not satisfy this"));
  if (file_system::check_if_file_exists("./Unit.IO.H5.FileErrorNotH5.h5ab")) {
    file_system::rm("./Unit.IO.H5.FileErrorNotH5.h5ab", true);
  }

  CHECK_THROWS_WITH(
      []() {
        const std::string file_name = "./Unit.IO.H5.FileErrorFileNotExist.h5";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        h5::H5File<h5::AccessType::ReadOnly> my_file(file_name);
      }(),
      Catch::Contains(
          "Trying to open the file './Unit.IO.H5.FileErrorFileNotExist.h5'"));
  if (file_system::check_if_file_exists(
          "./Unit.IO.H5.FileErrorFileNotExist.h5")) {
    file_system::rm("./Unit.IO.H5.FileErrorFileNotExist.h5", true);
  }

  CHECK_THROWS_WITH(
      []() {
        const std::string file_name =
            "./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        { h5::H5File<h5::AccessType::ReadWrite> my_file(file_name); }
        h5::H5File<h5::AccessType::ReadOnly> my_file(file_name, true);
      }(),
      Catch::Contains(
          "Cannot append to a file opened in read-only mode. File name is: "
          "./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5"));
  if (file_system::check_if_file_exists(
          "./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5")) {
    file_system::rm("./Unit.IO.H5.FileErrorCannotAppendReadOnly.h5", true);
  }

  CHECK_THROWS_WITH(
      []() {
        const std::string file_name = "./Unit.IO.H5.FileErrorExists.h5";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        // Need to close file opened by my_file otherwise we get the error
        // pure virtual method called
        // terminate called recursively
        {
          // [h5file_readwrite_file]
          h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
          // [h5file_readwrite_file]
        }
        h5::H5File<h5::AccessType::ReadWrite> my_file_2(file_name);
      }(),
      Catch::Contains(
          "File './Unit.IO.H5.FileErrorExists.h5' already exists and we are "
          "not allowed to append. To reduce the risk of accidental deletion "
          "you must explicitly delete the file first using the file_system "
          "library in SpECTRE or through your shell."));
  if (file_system::check_if_file_exists("./Unit.IO.H5.FileErrorExists.h5")) {
    file_system::rm("./Unit.IO.H5.FileErrorExists.h5", true);
  }

  CHECK_THROWS_WITH(
      []() {
        const std::string file_name =
            "./Unit.IO.H5.FileErrorObjectNotExistConst.h5";
        if (file_system::check_if_file_exists(file_name)) {
          file_system::rm(file_name, true);
        }
        const h5::H5File<h5::AccessType::ReadWrite> my_file(file_name);
        my_file.get<h5::Header>("/Dummy").get_header();
      }(),
      Catch::Contains(
          "Cannot open the object '/Dummy.hdr' because it does not exist."));
  if (file_system::check_if_file_exists(
          "./Unit.IO.H5.FileErrorObjectNotExistConst.h5")) {
    file_system::rm("./Unit.IO.H5.FileErrorObjectNotExistConst.h5", true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.H5.File", "[Unit][IO][H5]") {
  test_access_type();
  test_core_functionality();
  test_error_messages();
  test_file_move();
}
