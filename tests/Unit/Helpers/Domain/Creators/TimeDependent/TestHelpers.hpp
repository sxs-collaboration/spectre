// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace TestHelpers::domain::creators {
void write_volume_data(
    const std::string& filename, const std::string& subfile_name, double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time = {});
}  // namespace TestHelpers::domain::creators
