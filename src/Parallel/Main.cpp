// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Main.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <regex>
#include <string>
#include <tuple>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/FileSystem.hpp"

namespace Parallel::detail {
std::tuple<std::string, std::string, size_t> checkpoints_dir_prefix_pad() {
  const std::string checkpoints_dir = "Checkpoints";
  const std::string prefix = "Checkpoint_";
  constexpr size_t pad = 4;
  return std::make_tuple(checkpoints_dir, prefix, pad);
}

std::string next_checkpoint_dir(const size_t checkpoint_dir_counter) {
  const auto [checkpoints_dir, prefix, pad] =
      detail::checkpoints_dir_prefix_pad();
  const std::string counter = std::to_string(checkpoint_dir_counter);
  const std::string padded_counter =
      std::string(pad - counter.size(), '0').append(counter);
  std::string result = checkpoints_dir + "/" + prefix + padded_counter;
  if (file_system::check_if_dir_exists(result)) {
    ERROR("Can't write checkpoint: dir " + result + " already exists!");
  }
  return result;
}

void check_future_checkpoint_dirs_available(
    const size_t checkpoint_dir_counter) {
  const auto [checkpoints_dir, prefix, pad] =
      detail::checkpoints_dir_prefix_pad();
  if (not file_system::check_if_dir_exists(checkpoints_dir)) {
    return;
  }
  const auto next_checkpoint =
      detail::next_checkpoint_dir(checkpoint_dir_counter);

  // Find existing files with names that match the checkpoint dir name pattern
  const auto all_files = file_system::ls(checkpoints_dir);
  const std::regex re(prefix + "[0-9]{" + std::to_string(pad) + "}");
  std::vector<std::string> checkpoint_files;
  std::copy_if(all_files.begin(), all_files.end(),
               std::back_inserter(checkpoint_files),
               [&re](const std::string& s) { return std::regex_match(s, re); });

  // Using string comparison of filenames, check that all the files we found
  // are from older checkpoints, but not from future checkpoints
  const bool found_older_checkpoints_only = std::all_of(
      checkpoint_files.begin(), checkpoint_files.end(),
      [&next_checkpoint](const std::string& s) { return s < next_checkpoint; });
  if (not found_older_checkpoints_only) {
    ERROR(
        "Can't start run: found checkpoints that may be overwritten!\n"
        "Dirs from "
        << next_checkpoint << " onward must not exist.\n");
  }
}
}  // namespace Parallel::detail
