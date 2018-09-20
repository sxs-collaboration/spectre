// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Header.hpp"

#include <algorithm>
#include <regex>
#include <sstream>
#include <vector>

#include "IO/H5/Helpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
/// \cond HIDDEN_SYMOLS
Header::Header(const bool exists, detail::OpenGroup&& group,
               const hid_t location, const std::string& name)
    : group_(std::move(group)) {
  if (exists) {
    header_info_ =
        h5::read_rank1_attribute<std::string>(location, name + extension())[0];
  } else {
    std::vector<std::string> header_info{[]() {
      std::stringstream ss;
      ss << "#\n# File created on " << current_date_and_time() << "# ";
      auto build_info = info_from_build();
      ss << std::regex_replace(build_info, std::regex{"\n"}, "\n# ");
      return ss.str();
    }()};
    write_to_attribute(location, name + extension(), header_info);
    header_info_ = header_info[0];
  }
}
/// \endcond
}  // namespace h5
