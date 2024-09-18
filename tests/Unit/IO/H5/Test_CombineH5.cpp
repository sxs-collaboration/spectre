// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Cce.hpp"
#include "IO/H5/CombineH5.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
void test_single_file() {
  const std::string combined_filename{"CombinedSingle.h5"};
  const std::string original_filename{"OriginalFileName.h5"};
  if (file_system::check_if_file_exists(combined_filename)) {
    file_system::rm(combined_filename, true);
  }
  if (file_system::check_if_file_exists(original_filename)) {
    file_system::rm(original_filename, true);
  }

  const std::string subfile_name{"SubfileName"};
  const Matrix data{{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}};
  const std::vector<std::string> legend{"Time", "Data"};
  const uint32_t version = 4;

  {
    h5::H5File<h5::AccessType::ReadWrite> h5_file{original_filename};
    auto& dat_file = h5_file.insert<h5::Dat>(subfile_name, legend, version);
    dat_file.append(data);
  }

  h5::combine_h5_dat({original_filename}, combined_filename, Verbosity::Quiet);

  {
    const h5::H5File<h5::AccessType::ReadOnly> h5_file{combined_filename};
    const auto& dat_file = h5_file.get<h5::Dat>(subfile_name);
    CHECK(dat_file.get_version() == version);
    CHECK(dat_file.get_legend() == legend);
    CHECK(dat_file.get_data() == data);
  }

  if (file_system::check_if_file_exists(combined_filename)) {
    file_system::rm(combined_filename, true);
  }
  if (file_system::check_if_file_exists(original_filename)) {
    file_system::rm(original_filename, true);
  }
}

void test() {
  const std::string combined_filename{"MightyMorphinPowerRangers.h5"};
  if (file_system::check_if_file_exists(combined_filename)) {
    file_system::rm(combined_filename, true);
  }
  const std::vector<std::string> individual_filenames{
      "RedRanger.h5", "BlackRanger.h5", "BlueRanger.h5", "YellowRanger.h5"};
  const std::vector<std::string> subfile_names{"Subfile1", "Subfile2"};
  // All subfiles can just share the same legend. The data doesn't matter, only
  // the times for this test.
  const std::vector<std::string> legend{"Time", "Data"};
  const std::vector<std::vector<std::vector<double>>> data{
      // This file doesn't keep the last time
      {{0.0, 0.0}, {1.0, 1.0}, {2.0, 0.0}},
      // This file keeps all times, but is unordered
      {{2.1, 3.0}, {1.5, 2.0}},
      // This file only keeps the earliest time, but is unordered
      {{4.1, 0.0}, {3.0, 4.0}, {4.5, 0.0}},
      // This file keeps all times
      {{4.0, 5.0}, {5.5, 6.0}, {6.0, 7.0}}};

  const Matrix expected_data{{0.0, 0.0}, {1.0, 1.0}, {1.5, 2.0}, {2.1, 3.0},
                             {3.0, 4.0}, {4.0, 5.0}, {5.5, 6.0}, {6.0, 7.0}};

  // Write the individual files
  {
    for (size_t i = 0; i < individual_filenames.size(); i++) {
      const std::string& filename = individual_filenames[i];
      if (file_system::check_if_file_exists(filename)) {
        file_system::rm(filename, true);
      }
      h5::H5File<h5::AccessType::ReadWrite> h5_file{filename};
      for (const std::string& subfile_name : subfile_names) {
        auto& dat_file = h5_file.insert<h5::Dat>(subfile_name, legend);
        dat_file.append(data[i]);
        h5_file.close_current_object();
      }
    }
  }

  // Combine the H5 files
  h5::combine_h5_dat(individual_filenames, combined_filename,
                     Verbosity::Verbose);

  REQUIRE(file_system::check_if_file_exists(combined_filename));

  {
    const h5::H5File<h5::AccessType::ReadOnly> h5_file{combined_filename};
    for (const std::string& subfile_name : subfile_names) {
      CAPTURE(subfile_name);
      const auto& dat_file = h5_file.get<h5::Dat>(subfile_name);
      const Matrix dat_data = dat_file.get_data();
      CHECK(expected_data == dat_data);
      h5_file.close_current_object();
    }
  }

  if (file_system::check_if_file_exists(combined_filename)) {
    file_system::rm(combined_filename, true);
  }

  for (const std::string& filename : individual_filenames) {
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, true);
    }
  }
}

void test_errors() {
  const std::string fake_file{"FakeFile.h5"};
  const std::string error_filename_1{"CombineH5Error1.h5"};
  const std::string error_filename_2{"CombineH5Error2.h5"};
  CHECK_THROWS_WITH(
      h5::combine_h5_dat({}, fake_file),
      Catch::Matchers::ContainsSubstring("No H5 files to combine!"));

  const auto delete_files = [&]() {
    if (file_system::check_if_file_exists(error_filename_1)) {
      file_system::rm(error_filename_1, true);
    }
    if (file_system::check_if_file_exists(error_filename_2)) {
      file_system::rm(error_filename_2, true);
    }
    if (file_system::check_if_file_exists(fake_file)) {
      file_system::rm(fake_file, true);
    }
  };

  delete_files();

  {
    INFO("No dat files");
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_1};
      h5_file.insert<h5::Cce>("CceSubfile", 4);
    }
    CHECK_THROWS_WITH(
        h5::combine_h5_dat({error_filename_1}, fake_file),
        Catch::Matchers::ContainsSubstring("No dat files in H5 file"));
  }

  {
    INFO("No times in dat file");
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_1, true};
      h5_file.insert<h5::Dat>("DatSubfile",
                              std::vector<std::string>{"Time", "Blah"});
    }
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_2, true};
      h5_file.insert<h5::Dat>("DatSubfile",
                              std::vector<std::string>{"Time", "Blah"});
    }
    CHECK_THROWS_WITH(
        h5::combine_h5_dat({error_filename_1, error_filename_2}, fake_file),
        Catch::Matchers::ContainsSubstring("No times in dat file"));
  }

  delete_files();

  {
    INFO("Legends don't match");
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_1, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "Blah"});
      dat_file.append(std::vector{0.0, 0.0});
    }
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_2, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "DifferentBlah"});
      dat_file.append(std::vector{0.0, 0.0});
    }
    CHECK_THROWS_WITH(
        h5::combine_h5_dat({error_filename_1, error_filename_2}, fake_file),
        Catch::Matchers::ContainsSubstring("Legend of dat file") and
            Catch::Matchers::ContainsSubstring("doesn't match other H5 files"));
  }

  delete_files();

  {
    INFO("Versions don't match");
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_1, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "Blah"}, 0);
      dat_file.append(std::vector{0.0, 0.0});
    }
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_2, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "Blah"}, 1);
      dat_file.append(std::vector{0.0, 0.0});
    }
    CHECK_THROWS_WITH(
        h5::combine_h5_dat({error_filename_1, error_filename_2}, fake_file),
        Catch::Matchers::ContainsSubstring("Version of dat file") and
            Catch::Matchers::ContainsSubstring("doesn't match other H5 files"));
  }

  delete_files();

  {
    INFO("Non monotonically increasing H5 files");
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_1, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "Blah"}, 0);
      dat_file.append(std::vector{1.0, 0.0});
    }
    {
      h5::H5File<h5::AccessType::ReadWrite> h5_file{error_filename_2, true};
      auto& dat_file = h5_file.try_insert<h5::Dat>(
          "DatSubfile", std::vector<std::string>{"Time", "Blah"}, 0);
      dat_file.append(std::vector{0.0, 0.0});
    }
    CHECK_THROWS_WITH(
        h5::combine_h5_dat({error_filename_1, error_filename_2}, fake_file),
        Catch::Matchers::ContainsSubstring(
            "are not monotonically increasing in their first "
            "times for dat file"));
  }
}
}  // namespace

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.IO.H5.CombineH5", "[Unit][IO][H5]") {
  test_single_file();
  test_errors();
  test();
}
