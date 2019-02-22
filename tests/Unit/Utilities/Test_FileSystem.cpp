// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <fstream>
#include <string>

#include "Utilities/FileSystem.hpp"

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_parent_path",
                  "[Unit][Utilities]") {
  /// [get_parent_path]
  CHECK("/test/path/to/dir"s ==
        file_system::get_parent_path("/test/path/to/dir/dummy.txt"));
  CHECK("/test/path/to"s == file_system::get_parent_path("/test/path/to/dir/"));
  CHECK("/"s == file_system::get_parent_path("/"));
  CHECK("path/to/dir"s ==
        file_system::get_parent_path("path/to/dir/dummy.txt"));
  CHECK("/usr"s == file_system::get_parent_path("/usr/lib/"));
  CHECK("/"s == file_system::get_parent_path("/usr"));
  CHECK("."s == file_system::get_parent_path("usr"));
  CHECK("."s == file_system::get_parent_path(".."));
  CHECK("."s == file_system::get_parent_path(""));
  /// [get_parent_path]
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_file_name",
                  "[Unit][Utilities]") {
  /// [get_file_name]
  CHECK("dummy.txt"s ==
        file_system::get_file_name("/test/path/to/dir/dummy.txt"));
  CHECK(".dummy.txt"s ==
        file_system::get_file_name("/test/path/to/dir/.dummy.txt"));
  CHECK("dummy.txt"s == file_system::get_file_name("./dummy.txt"));
  CHECK("dummy.txt"s == file_system::get_file_name("../dummy.txt"));
  CHECK(".dummy.txt"s == file_system::get_file_name(".dummy.txt"));
  CHECK("dummy.txt"s == file_system::get_file_name("dummy.txt"));
  CHECK(".dummy"s == file_system::get_file_name(".dummy"));
  /// [get_file_name]
}

// [[OutputRegex, Failed to find a file in the given path: '/']]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_file_name_error",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  file_system::get_file_name("/");
}

// [[OutputRegex, Received an empty path]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_file_name_empty_path",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  static_cast<void>(file_system::get_file_name(""));
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_absolute_path",
                  "[Unit][Utilities]") {
  CHECK(file_system::cwd() == file_system::get_absolute_path("./"));
}

// [[OutputRegex, Failed to convert to absolute path because one of the path
// components does not exist. Relative path is]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.get_absolute_path_nonexistent",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  static_cast<void>(
      file_system::get_absolute_path("./get_absolute_path_nonexistent/"));
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.check_if_exists",
                  "[Unit][Utilities]") {
  CHECK(file_system::check_if_dir_exists("./"));
  std::fstream file("check_if_exists.txt", file.out);
  file.close();
  CHECK(file_system::check_if_file_exists("./check_if_exists.txt"));
  CHECK(0 == file_system::file_size("./check_if_exists.txt"));

  file = std::fstream("check_if_exists.txt", file.out);
  file << "Write something";
  file.close();
  CHECK(0 < file_system::file_size("./check_if_exists.txt"));

  file_system::rm("./check_if_exists.txt", false);
  CHECK_FALSE(file_system::check_if_file_exists("./check_if_exists.txt"));
}

// [[OutputRegex, Failed to check if path points to a file because the path is
// invalid.]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.is_file_error",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  CHECK(file_system::is_file("./is_file_error"));
}

// [[OutputRegex, Cannot get size of file './file_size_error.txt' because it
// cannot be accessed. Either it does not exist or you do not have the
// appropriate permissions.]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.file_size_error",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  CHECK(file_system::file_size("./file_size_error.txt"));
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.create_and_rm_directory",
                  "[Unit][Utilities]") {
  const std::string dir_one(
      "./create_and_rm_directory/nested///nested2/nested3///");
  file_system::create_directory(dir_one);
  CHECK(file_system::check_if_dir_exists(dir_one));
  const std::string dir_two("./create_and_rm_directory/nested/nested2/nested4");
  file_system::create_directory(dir_two);
  CHECK(file_system::check_if_dir_exists(dir_two));
  std::fstream file(
      "./create_and_rm_directory//nested/nested2/nested4/check_if_exists.txt",
      file.out);
  file.close();
  // Check that creating an existing directory does nothing
  file_system::create_directory(dir_two);
  CHECK(file_system::check_if_dir_exists(dir_two));
  CHECK(file_system::check_if_file_exists(dir_two + "/check_if_exists.txt"s));
  file_system::rm("./create_and_rm_directory"s, true);
  CHECK_FALSE(file_system::check_if_dir_exists("./create_and_rm_directory"s));
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.create_and_rm_empty_directory",
                  "[Unit][Utilities]") {
  const std::string dir_name("./create_and_rm_empty_directory");
  file_system::create_directory(dir_name);
  CHECK(file_system::check_if_dir_exists(dir_name));
  file_system::rm(dir_name, false);
  CHECK_FALSE(file_system::check_if_dir_exists(dir_name));
}

// [[OutputRegex, Cannot create a directory that has no name]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.create_dir_error_cannot_be_empty",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  file_system::create_directory(""s);
}

SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.create_dir_root",
                  "[Unit][Utilities]") {
  file_system::create_directory("/"s);
  CHECK(file_system::check_if_dir_exists("/"s));
}

// [[OutputRegex, Could not delete file './rm_error_not_empty' because the
// directory is not empty]]
SPECTRE_TEST_CASE("Unit.Utilities.FileSystem.rm_error_not_empty",
                  "[Unit][Utilities]") {
  ERROR_TEST();
  const std::string dir_name("./rm_error_not_empty");
  file_system::create_directory(dir_name);
  CHECK(file_system::check_if_dir_exists(dir_name));
  std::fstream file("./rm_error_not_empty/cause_error_in_rm.txt", file.out);
  file.close();
  file_system::rm("./rm_error_not_empty"s, false);
}
