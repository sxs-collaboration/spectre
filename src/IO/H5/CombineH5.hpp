// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

namespace h5 {

void combine_h5(const std::vector<std::string>& file_names,
                const std::string& subfile_name, const std::string& output,
                const bool check_src = true);

}  // namespace h5
