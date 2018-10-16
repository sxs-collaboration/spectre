// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Header.hpp"

#include <algorithm>
#include <regex>
#include <string>
#include <vector>

#include "IO/H5/Helpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/Formaline.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
/// \cond HIDDEN_SYMOLS
Header::Header(const bool exists, detail::OpenGroup&& group,
               const hid_t location, const std::string& name)
    : group_(std::move(group)) {
  if (exists) {
    header_info_ =
        h5::read_rank1_attribute<std::string>(location, name + extension())[0];
    if (header_info_.find(printenv_delimiter_) != std::string::npos) {
      const auto printenv_location =
          header_info_.find(printenv_delimiter_) + printenv_delimiter_.size();
      const auto library_versions_location =
          header_info_.find(library_versions_delimiter_);
      environment_variables_ = header_info_.substr(
          printenv_location, library_versions_location - printenv_location);
      library_versions_ = header_info_.substr(
          library_versions_location + library_versions_delimiter_.size());
      header_info_.erase(printenv_location - printenv_delimiter_.size());
    }

    else {
      //If we cannot find the Formaline delimiter in the file then the file
      //was written without Formaline support and so we fill in fake info.
      environment_variables_ =
          "Formaline was not supported when file was written";
      library_versions_ = "Formaline was not supported when file was written";
    }
  } else {
    auto build_info = info_from_build();
    environment_variables_ = formaline::get_environment_variables();
    library_versions_ = formaline::get_library_versions();
    header_info_ = MakeString{}
                   << "#\n# File created on " << current_date_and_time() << "# "
                   << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
    write_to_attribute(
        location, name + extension(),
        std::vector<std::string>{
            MakeString{} << header_info_ << printenv_delimiter_
                         << environment_variables_
                         << library_versions_delimiter_ << library_versions_});
  }
}

std::string Header::get_env_variables() const noexcept {
  return environment_variables_;
}

std::string Header::get_library_versions() const noexcept {
  return library_versions_;
}

const std::string Header::printenv_delimiter_{
    "############### printenv ###############\n"};
const std::string Header::library_versions_delimiter_{
    "############### library versions ###############\n"};
/// \endcond
}  // namespace h5
