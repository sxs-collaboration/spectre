// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"

namespace {
void create_file(const char* const file) { std::fstream(file, std::ios::out); }

void test() {
  {
    INFO("get_parent_path");
    // [get_parent_path]
    CHECK("/test/path/to/dir"s ==
          file_system::get_parent_path("/test/path/to/dir/dummy.txt"));
    CHECK("/test/path/to"s ==
          file_system::get_parent_path("/test/path/to/dir/"));
    CHECK("/"s == file_system::get_parent_path("/"));
    CHECK("path/to/dir"s ==
          file_system::get_parent_path("path/to/dir/dummy.txt"));
    CHECK("/usr"s == file_system::get_parent_path("/usr/lib/"));
    CHECK("/"s == file_system::get_parent_path("/usr"));
    CHECK("."s == file_system::get_parent_path("usr"));
    CHECK("."s == file_system::get_parent_path(".."));
    CHECK("."s == file_system::get_parent_path(""));
    // [get_parent_path]
  }
  {
    INFO("get_file_name");
    // [get_file_name]
    CHECK("dummy.txt"s ==
          file_system::get_file_name("/test/path/to/dir/dummy.txt"));
    CHECK(".dummy.txt"s ==
          file_system::get_file_name("/test/path/to/dir/.dummy.txt"));
    CHECK("dummy.txt"s == file_system::get_file_name("./dummy.txt"));
    CHECK("dummy.txt"s == file_system::get_file_name("../dummy.txt"));
    CHECK(".dummy.txt"s == file_system::get_file_name(".dummy.txt"));
    CHECK("dummy.txt"s == file_system::get_file_name("dummy.txt"));
    CHECK(".dummy"s == file_system::get_file_name(".dummy"));
    // [get_file_name]
  }
  {
    INFO("get_absolute_path");
    CHECK(file_system::cwd() == file_system::get_absolute_path("./"));
  }
  {
    INFO("check_if_exists");
    CHECK(file_system::check_if_dir_exists("./"));
    create_file("check_if_exists.txt");
    CHECK(file_system::check_if_file_exists("./check_if_exists.txt"));
    CHECK(0 == file_system::file_size("./check_if_exists.txt"));

    {
      std::fstream file("check_if_exists.txt", file.out);
      file << "Write something";
    }
    CHECK(0 < file_system::file_size("./check_if_exists.txt"));

    file_system::rm("./check_if_exists.txt", false);
    CHECK_FALSE(file_system::check_if_file_exists("./check_if_exists.txt"));
  }
  {
    INFO("create_and_rm_directory");
    const std::string dir_one(
        "./create_and_rm_directory/nested///nested2/nested3///");
    file_system::create_directory(dir_one);
    CHECK(file_system::check_if_dir_exists(dir_one));
    const std::string dir_two(
        "./create_and_rm_directory/nested/nested2/nested4");
    file_system::create_directory(dir_two);
    CHECK(file_system::check_if_dir_exists(dir_two));
    create_file("./create_and_rm_directory//nested/nested2/nested4/"
                "check_if_exists.txt");
    // Check that creating an existing directory does nothing
    file_system::create_directory(dir_two);
    CHECK(file_system::check_if_dir_exists(dir_two));
    CHECK(file_system::check_if_file_exists(dir_two + "/check_if_exists.txt"s));
    file_system::rm("./create_and_rm_directory"s, true);
    CHECK_FALSE(file_system::check_if_dir_exists("./create_and_rm_directory"s));
  }
  {
    INFO("create_and_rm_empty_directory");
    const std::string dir_name("./create_and_rm_empty_directory");
    file_system::create_directory(dir_name);
    CHECK(file_system::check_if_dir_exists(dir_name));
    file_system::rm(dir_name, false);
    CHECK_FALSE(file_system::check_if_dir_exists(dir_name));
  }
  {
    INFO("create_dir_root");
    file_system::create_directory("/"s);
    CHECK(file_system::check_if_dir_exists("/"s));
  }
  {
    INFO("glob");
    create_file("glob1.txt");
    create_file("glob2.txt");
    CHECK(file_system::glob("glob*.txt") ==
          std::vector<std::string>{"glob1.txt", "glob2.txt"});
    file_system::rm("glob1.txt", false);
    file_system::rm("glob2.txt", false);
  }
  {
    INFO("ls");
    file_system::create_directory("ls_test"s);
    create_file("ls_test/file1");
    create_file("ls_test/file2");
    create_file("ls_test/file3");
    file_system::create_directory("ls_test/dir1"s);
    create_file("ls_test/dir1/nested_file");
    std::vector<std::string> expected_list{"file1", "file2", "file3", "dir1"};
    auto list = file_system::ls("ls_test");
    alg::sort(list);
    alg::sort(expected_list);
    CHECK(list == expected_list);
  }
}

void trigger_nonexistent_absolute_path() {
  static_cast<void>(
      file_system::get_absolute_path("./get_absolute_path_nonexistent/"));
}

void trigger_rm_error_not_empty() {
  const std::string dir_name("./rm_error_not_empty");
  file_system::create_directory(dir_name);
  CHECK(file_system::check_if_dir_exists(dir_name));
  std::fstream file("./rm_error_not_empty/cause_error_in_rm.txt", file.out);
  file.close();
  file_system::rm("./rm_error_not_empty"s, false);
}

void test_errors() {
  CHECK_THROWS_WITH(
      file_system::get_file_name("/"),
      Catch::Contains("Failed to find a file in the given path: '/'"));
  CHECK_THROWS_WITH(file_system::get_file_name(""),
                    Catch::Contains("Received an empty path"));
  CHECK_THROWS_WITH(trigger_nonexistent_absolute_path(),
                    Catch::Contains("No such file or directory"));
  CHECK_THROWS_WITH(
      file_system::file_size("./file_size_error.txt"),
      Catch::Contains("Cannot get size of file './file_size_error.txt' because "
                      "it cannot be accessed. Either it does not exist or you "
                      "do not have the appropriate permissions."));
  CHECK_THROWS_WITH(trigger_rm_error_not_empty(),
                    Catch::Contains("remove: Directory not empty"));
  CHECK_THROWS_WITH(file_system::is_file("./is_file_error"),
                    Catch::Contains("Failed to check if path points to a file "
                                    "because the path is invalid"));
  CHECK_THROWS_WITH(
      file_system::create_directory(""s),
      Catch::Contains("Cannot create a directory that has no name"));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem", "[Unit][Utilities]") {
  // This is a macro to report locations better and to ensure that
  // creating the argument can't modify errno before we save it.
#define SYS_ERROR(description)                                 \
  do {                                                         \
    const int errno_save = errno;                              \
    ERROR(description << " failed: " << strerror(errno_save)); \
  } while (false)

  // Run the test in a temporary directory so we have a known starting
  // state for the filesystem.  This also simplifies cleanup.
  const int original_directory = open(".", O_RDONLY);
  if (original_directory == -1) {
    SYS_ERROR("opendir(.)");
  }
  char scratch_directory[] = "Test_FileSystem_scratch-XXXXXX";
  if (mkdtemp(scratch_directory) == nullptr) {
    // scratch_directory may have been modified, so reporting it in
    // the error would be misleading.
    SYS_ERROR("mkdtemp");
  }
  const auto cleanup = [&]() {
    if (fchdir(original_directory) != 0) {
      SYS_ERROR("fchdir");
    }
    close(original_directory);
    file_system::rm(scratch_directory, true);
  };

  try {
    if (chdir(scratch_directory) != 0) {
      SYS_ERROR("chrdir(" << scratch_directory << ")");
    }
    test();
    test_errors();
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
}
