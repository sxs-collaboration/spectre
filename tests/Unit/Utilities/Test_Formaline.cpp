// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>
#include <vector>

#include "Utilities/FileSystem.hpp"
#include "Utilities/Formaline.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.Formaline", "[Unit][Utilities]") {
#ifndef __APPLE__
  SECTION("check archive size") {
    const auto archive = formaline::get_archive();
    // The archive should be at least 1.7MB since it is ~1.73MB as of
    // 02/14/2019.
    CHECK(archive.size() > 1740 * 1024);
  }
  SECTION("check paths") {
    const auto paths = formaline::get_paths();
    CHECK(paths.find("PATH=") != std::string::npos);
    CHECK(paths.find("CPATH=") != std::string::npos);
    CHECK(paths.find("LD_LIBRARY_PATH=") != std::string::npos);
    CHECK(paths.find("LIBRARY_PATH=") != std::string::npos);
    CHECK(paths.find("CMAKE_PREFIX_PATH=") != std::string::npos);
  }
  SECTION("check writing archive to file") {
    const std::string basename = "./FormalineTestArchive";
    const std::string filename = basename + ".tar.gz";
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, false);
    }
    formaline::write_to_file(basename);
    CHECK(file_system::check_if_file_exists(filename));
    if (file_system::check_if_file_exists(filename)) {
      file_system::rm(filename, false);
    }
  }
#else
  // Need to do at least 1 check otherwise Catch will fail the test.
  CHECK(true);
#endif  // defined(__APPLE__)
}
