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
#include <dirent.h>
#include <libgen.h>
#include <memory>
#include <regex>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>  // IWYU pragma: keep
#include <thread>
#include <unistd.h>

#include "ErrorHandling/Error.hpp"
#include "Parallel/Printf.hpp"

// IWYU asks to add <fcntl.h> when it's not there and remove it when it's there.
// IWYU pragma: no_include <fcntl.h>

namespace file_system {

std::string get_parent_path(const std::string& path) {
  std::vector<char> file_path(path.length() + 1);
  strncpy(file_path.data(), path.c_str(), path.length() + 1);
  // The pointer from ::dirname is not freed since it aliases with file_path
  char* parent_dir_name = ::dirname(file_path.data());
  std::string return_name(parent_dir_name);
  return return_name;
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

std::string get_absolute_path(const std::string& rel_path) {
  // clang-tidy: do not manually manage memory
  auto deleter = [](char* p) { free(p); }; // NOLINT
  std::unique_ptr<char, decltype(deleter)> name(
      realpath(rel_path.c_str(), nullptr), deleter);
  if (nullptr == name) {
    if (ENAMETOOLONG == errno) {
      // LCOV_EXCL_START
      ERROR(
          "Failed to convert to absolute path because the resulting name is "
          "too long. Relative path is: '"
          << rel_path << "'.");
      // LCOV_EXCL_STOP
    } else if (EACCES == errno) {
      // LCOV_EXCL_START
      ERROR(
          "Failed to convert to absolute path because one of the components of "
          "the path does not have proper read access. Relative path is: '"
          << rel_path << "'.");
      // LCOV_EXCL_STOP
    } else if (ENOENT == errno) {
      ERROR(
          "Failed to convert to absolute path because one of the path "
          "components does not exist. Relative path is: '"
          << rel_path << "'.");
      // LCOV_EXCL_START
    } else if (ELOOP == errno) {
      ERROR(
          "Failed to convert to absolute path because the maximum number of "
          "symlinks was in the path. Relative path is: '"
          << rel_path << "'.");
    }
    const auto local_errno = errno;
    ERROR("Failed to get absolute path for an unknown reason: " << local_errno);
    // LCOV_EXCL_STOP
  }
  return name.get();
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

  // This flag makes sure that we don't try to create the parent directory
  // more than once (which happens if we get stuck in an infinite loop)
  bool tried_to_create_parent_dir = false;
  // Try multiple times for slow filesystems
  for (size_t number_of_failures = 0; number_of_failures < num_tries;
       ++number_of_failures) {
    // Make the directory (returns errno==EEXIST if it already exists)
    const int status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
    const int the_errno = errno;

    // Note: if successful (or if dir already exists), don't wait just break.
    if (status == 0 or the_errno == EEXIST) {
      return;
    }

    // if a part of the path is missing, try walking up the path and create it
    if (status == -1 and the_errno == ENOENT) {
      if (tried_to_create_parent_dir) {
        // we already tried to create this parent directory, so we must
        // be stuck in an infinite loop, there's no point in trying further
        // LCOV_EXCL_START
        ERROR("Got stuck in an infinite loop while trying to create '"
              << dir << "'.\n");
        // LCOV_EXCL_STOP
      }
      create_directory(file_system::get_parent_path(dir), wait_time, num_tries);
      tried_to_create_parent_dir = true;
    } else {
      // Ok, we failed. Try again after waiting
      // LCOV_EXCL_START
      Parallel::printf(
          "create_directory: mkdir(%s) failed %d time(s). Error: %d\n", dir,
          number_of_failures + 1, the_errno);
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
  struct stat buf {};
  // stat returns 0 if the operation is successful (thing exists)
  return 0 == stat(dir.c_str(), &buf) and S_ISDIR(buf.st_mode);
}

bool check_if_file_exists(const std::string& file) {
  struct stat buf {};
  // stat returns 0 if the operation is successful (thing exists)
  return 0 == stat(file.c_str(), &buf) and S_ISREG(buf.st_mode);
}

bool is_file(const std::string& path) {
  struct stat buf {};
  // stat returns 0 if the operation is successful (thing exists)
  if (0 != stat(path.c_str(), &buf)) {
    ERROR(
        "Failed to check if path points to a file because the path is invalid. "
        "Given path is: "
        << path);
  }
  return S_ISREG(buf.st_mode);
}

size_t file_size(const std::string& file) {
  struct stat buf {};
  // stat returns 0 if the operation is successful (thing exists)
  if (0 != stat(file.c_str(), &buf)) {
    ERROR("Cannot get size of file '"
          << file << "' because it cannot be accessed. Either it does not "
                     "exist or you do not have the appropriate permissions.");
  }
  return static_cast<size_t>(buf.st_size);
}

std::string cwd() {
  double wait_time = 1;
  // clang-tidy: do not manually manage memory
  auto deleter = [](char* p) { free(p); }; // NOLINT
  std::unique_ptr<char, decltype(deleter)> the_cwd(nullptr, deleter);
  the_cwd.reset(getcwd(the_cwd.get(), 0));
  while (the_cwd == nullptr) {
    // It's not clear how to test this code since we can't make the file system
    // be slow
    // LCOV_EXCL_START
    std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
    wait_time += 10;
    the_cwd.reset(getcwd(the_cwd.get(), 0));
    if (wait_time > 61) {
      ERROR(
          "Could not get the current directory. This is typically related to "
          "filesystem issues.");
    }
    // LCOV_EXCL_STOP
  }
  return the_cwd.get();
}

std::vector<std::string> ls(const std::string& dir_name) {
  std::vector<std::string> contents;
  DIR* dir = opendir(dir_name.c_str());
  double wait_time = 1;
  while (dir == nullptr) {
    // LCOV_EXCL_START
    std::this_thread::sleep_for(std::chrono::duration<double>(wait_time));
    wait_time += 10;
    if (wait_time > 61) {
      ERROR("Failed to open directory '" << dir_name << "'");
    }
    dir = opendir(dir_name.c_str());
    // LCOV_EXCL_STOP
  }
  struct dirent* file;
  // readdir returns next file in dir
  while (nullptr != (file = readdir(dir))) {
    contents.emplace_back(static_cast<char*>(file->d_name));
  }
  closedir(dir);
  return contents;
}

void rm(const std::string& path, bool recursive) {
  if (recursive) {
    if (not is_file(path)) {
      auto contents = ls(path);
      contents.erase(std::find(contents.begin(), contents.end(), "."s));
      contents.erase(std::find(contents.begin(), contents.end(), ".."s));
      std::transform(
          contents.begin(), contents.end(), contents.begin(),
          [&path](std::string& element) { return path + "/" + element; });
      for (const auto& files : contents) {
        rm(files, recursive);
      }
    }
  }
  if (0 != remove(path.c_str())) {
    // LCOV_EXCL_START
    if (EACCES == errno) {
      ERROR("Could not delete file '" << path
                                      << "' because of incorrect permissions.");
    } else if (EBUSY == errno) {
      ERROR("Could not delete file '" << path << "' because it is busy.");
    } else if (ENOENT == errno) {
      ERROR("Could not delete file '" << path
                                      << "' because it does not exist.");
    } else if (EROFS == errno) {
      ERROR("Could not delete file '"
            << path << "' because it is on a read-only filesystem.");
    } else if (ENOTEMPTY == errno or EEXIST == errno) {
      ERROR("Could not delete file '"
            << path << "' because the directory is not empty");
    }
    ERROR("Could not delete file '" << path
                                    << "' because an unknown error occurred.");
    // LCOV_EXCL_STOP
  }
}
}  // namespace file_system
