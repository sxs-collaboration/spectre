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
Header::Header(const bool exists, detail::OpenGroup&& group,
               const hid_t location, const std::string& name)
    : group_(std::move(group)) {
  if (exists) {
    header_info_ =
        h5::read_rank1_attribute<std::string>(location, name + extension())[0];
    if (header_info_.find(printenv_delimiter_) != std::string::npos) {
      const auto printenv_location =
          header_info_.find(printenv_delimiter_) + printenv_delimiter_.size();
      const auto build_info_location = header_info_.find(build_info_delimiter_);
      environment_variables_ = header_info_.substr(
          printenv_location, build_info_location - printenv_location);
      build_info_ = header_info_.substr(
          build_info_location + build_info_delimiter_.size());
      header_info_.erase(printenv_location - printenv_delimiter_.size());
    }

    else {
      // If we cannot find the Formaline delimiter in the file then the file
      // was written without Formaline support and so we fill in fake info.
      environment_variables_ =
          "Formaline was not supported when file was written";
      build_info_ = "Formaline was not supported when file was written";
    }
  } else {
    auto build_info = info_from_build();
    environment_variables_ = formaline::get_environment_variables();
    build_info_ = formaline::get_build_info();
    header_info_ = MakeString{}
                   << "#\n# File created on " << current_date_and_time() << "# "
                   << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
    write_to_attribute(
        location, name + extension(),
        std::vector<std::string>{
            MakeString{} << header_info_ << printenv_delimiter_
                         << environment_variables_
                         << build_info_delimiter_ << build_info_});
  }
}

std::string Header::get_env_variables() const { return environment_variables_; }

std::string Header::get_build_info() const { return build_info_; }

const std::string Header::printenv_delimiter_{
    "############### printenv ###############\n"};
const std::string Header::build_info_delimiter_{
    "############### library versions ###############\n"};
}  // namespace h5
