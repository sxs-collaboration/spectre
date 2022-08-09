// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/FileSystem.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <glob.h>
#include <libgen.h>
#include <memory>
#include <regex>
#include <sstream>
#include <thread>

#include "Parallel/Printf.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

// IWYU asks to add <fcntl.h> when it's not there and remove it when it's there.
// IWYU pragma: no_include <fcntl.h>

namespace file_system {

void copy(const std::string& from, const std::string& to) {
  std::filesystem::copy(from, to);
}

std::string cwd() {
  double wait_time = 1;
  std::string current_path = std::filesystem::current_path();
  while (current_path.empty()) {
    // It's not clear how to test this code since we can't make the file system
    // be slow
    // LCOV_EXCL_START
    std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
    wait_time += 10;
    try {
      current_path = std::filesystem::current_path();
    } catch (const std::exception& e) {
      if (wait_time > 61) {
        ERROR(
            "Could not get the current directory. This is typically related to "
            "filesystem issues. Exception message: "
            << e.what());
      }
    }
    // LCOV_EXCL_STOP
  }
  return current_path;
}

void create_directory(const std::string& dir, const double wait_time,
                      const size_t num_tries) {
  // each time we fail to create a directory, we increase the wait_time by this
  // factor
  static constexpr double wait_time_increase_factor = 1.1;
  if (dir.empty()) {
    ERROR("Cannot create a directory that has no name");
  }
  if (std::string::npos == dir.find_first_not_of('/')) {
    return;  // trying to create directory '/'
  }

  // Try multiple times for slow filesystems
  for (size_t number_of_failures = 0; number_of_failures < num_tries;
       ++number_of_failures) {
    try {
      std::filesystem::create_directories(dir);
      return;
    } catch (const std::exception& e) {
      // LCOV_EXCL_START
      Parallel::printf(
          "create_directory: mkdir(%s) failed %d time(s). Error: %s\n", dir,
          number_of_failures + 1, e.what());
      std::this_thread::sleep_for(std::chrono::duration<double>(
          wait_time *
          std::pow(wait_time_increase_factor, number_of_failures + 1)));
      // LCOV_EXCL_STOP
    }
  }
  // LCOV_EXCL_START
  ERROR("Unable to mkdir '" << dir << "'. Giving up after " << num_tries
                            << " tries\n");
  // LCOV_EXCL_STOP
}

bool check_if_dir_exists(const std::string& dir) {
  return std::filesystem::exists(dir) and std::filesystem::is_directory(dir);
}

bool check_if_file_exists(const std::string& file) {
  return std::filesystem::exists(file) and
         std::filesystem::is_regular_file(file);
}

size_t file_size(const std::string& file) {
  if (not check_if_file_exists(file)) {
    ERROR("Cannot get size of file '"
          << file
          << "' because it cannot be accessed. Either it does not "
             "exist or you do not have the appropriate permissions.");
  }
  return static_cast<size_t>(std::filesystem::file_size(file));
}

std::string get_absolute_path(const std::string& rel_path) {
  return std::filesystem::canonical(rel_path);
}

std::string get_file_name(const std::string& file_path) {
  if (file_path.empty()) {
    ERROR("Received an empty path");
  }
  if (file_path.find('/') == std::string::npos) {
    // Handle file names such as 'dummy.txt' or '.dummy.txt'
    return file_path;
  }
  std::smatch match{};
  std::regex file_name_pattern{R"(^.*/([^/]+))"};
  auto regex_matched = std::regex_search(file_path, match, file_name_pattern);
  if (not regex_matched) {
    ERROR("Failed to find a file in the given path: '" << file_path << "'");
  }
  return match[1];
}

std::string get_parent_path(const std::string& path) {
  std::vector<char> file_path(path.length() + 1);
  strncpy(file_path.data(), path.c_str(), path.length() + 1);
  // The pointer from ::dirname is not freed since it aliases with file_path
  char* parent_dir_name = ::dirname(file_path.data());
  std::string return_name(parent_dir_name);
  return return_name;
}

std::vector<std::string> glob(const std::string& pattern) {
  glob_t buffer;
  const int return_value =
      ::glob(pattern.c_str(), GLOB_TILDE, nullptr, &buffer);
  if (return_value != 0) {
    ERROR("Unable to resolve glob '" + pattern + "': " + std::strerror(errno));
  }
  std::vector<std::string> file_names(
      buffer.gl_pathv,
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      buffer.gl_pathv + buffer.gl_pathc);
  globfree(&buffer);
  return file_names;
}

bool is_file(const std::string& path) {
  if (not std::filesystem::exists(path)) {
    ERROR(
        "Failed to check if path points to a file because the path is invalid. "
        "Given path is: "
        << path);
  }
  return std::filesystem::is_regular_file(path);
}

std::vector<std::string> ls(const std::string& dir_name) {
  std::vector<std::string> contents;
  for (auto const& dir_entry : std::filesystem::directory_iterator{dir_name}) {
    contents.push_back(dir_entry.path().filename());
  }
  return contents;
}

void rm(const std::string& path, bool recursive) {
  if (recursive) {
    std::filesystem::remove_all(path);
  } else {
    std::filesystem::remove(path);
  }
}
}  // namespace file_system
