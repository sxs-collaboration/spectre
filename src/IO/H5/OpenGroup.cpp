// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/OpenGroup.hpp"

#include <H5version.h>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <regex>
#include <sstream>

#include "ErrorHandling/Error.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Wrappers.hpp"

namespace {
std::vector<std::string> split_strings(const std::string& s,
                                       const bool match_multiple = true) {
  std::regex split_delim(match_multiple ? R"([^/]+)" : R"([^/])");
  auto words_begin = std::sregex_iterator(s.begin(), s.end(), split_delim);
  auto words_end = std::sregex_iterator();
  std::vector<std::string> split;
  split.reserve(static_cast<size_t>(std::distance(words_begin, words_end)));
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    split.emplace_back(i->str());
  }
  return split;
}
}  // namespace

namespace h5 {
namespace detail {
OpenGroup::OpenGroup(hid_t file_id, const std::string& group_name,
                     const h5::AccessType access_type) {
  const std::vector<std::string> path(split_strings(group_name));
  hid_t group_id = file_id;
  group_path_.reserve(path.size());
  for (const auto& current_group : path) {
    if (not current_group.empty()) {
      const auto status = H5Lexists(group_id, current_group.c_str(), 0);
      if (0 < status) {
        group_id = H5Gopen(group_id, current_group.c_str(), h5p_default());
      } else if (0 == status) {
        if (AccessType::ReadOnly == access_type) {
          ERROR("Cannot create group '" << current_group.c_str()
                                        << "' in path: " << group_name
                                        << " because the access is ReadOnly");
        }
        group_id = H5Gcreate(group_id, current_group.c_str(), h5p_default(),
                             h5p_default(), h5p_default());
      } else {
        ERROR("Failed to open the group '"
              << current_group
              << "' because the file_id passed in is invalid, or because the "
                 "group_id inside the OpenGroup constructor got corrupted. It "
                 "is most likely that the file_id is invalid.");
      }
      CHECK_H5(group_id, "Failed to open group '" << current_group << "'");
      group_path_.push_back(group_id);
    }
  }
  group_id_ = group_id;
}

/// \cond HIDDEN_SYMBOLS
OpenGroup::OpenGroup(OpenGroup&& rhs) noexcept {
  group_path_ = std::move(rhs.group_path_);
  // clang-tidy: moving trivial type has no effect: might change in future
  group_id_ = std::move(rhs.group_id_);  // NOLINT

  rhs.group_id_ = -1;
}

OpenGroup& OpenGroup::operator=(OpenGroup&& rhs) noexcept {
  if (group_id_ != -1) {
    for (auto rit = group_path_.rbegin(); rit != group_path_.rend(); ++rit) {
      H5Gclose(*rit);
    }
  }

  group_path_ = std::move(rhs.group_path_);
  // clang-tidy: moving trivial type has no effect: might change in future
  group_id_ = std::move(rhs.group_id_);  // NOLINT

  rhs.group_id_ = -1;
  return *this;
}

OpenGroup::~OpenGroup() {
  if (group_id_ != -1) {
    for (auto rit = group_path_.rbegin(); rit != group_path_.rend(); ++rit) {
      H5Gclose(*rit);
    }
    // Allows destructor to be run manually in tests without relying on HDF5
    // implementation
    group_id_ = -1;
  }
}
/// \endcond
}  // namespace detail
}  // namespace h5
