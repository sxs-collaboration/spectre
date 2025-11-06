// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Creators/TimeDependentOptions/FromVolumeFile.hpp"

#include <array>
#include <boost/math/quaternion.hpp>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionHelpers.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Options/Context.hpp"
#include "Options/ParseError.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace domain::creators::time_dependent_options {
namespace {
// Returns a clone of the FoT requested
std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> get_function_of_time(
    const std::string& function_of_time_name, const std::string& h5_filename,
    const std::string& subfile_name, const std::optional<double>& time) {
  const h5::H5File<h5::AccessType::ReadOnly> h5_file{h5_filename};
  const auto& vol_file = h5_file.get<h5::VolumeData>(subfile_name);
  std::optional<std::vector<char>> serialized_functions_of_time =
      vol_file.get_global_functions_of_time();
  if (not serialized_functions_of_time.has_value()) {
    ERROR_NO_TRACE(function_of_time_name
                   << ": There are no functions of time in the subfile "
                   << subfile_name << " of the H5 file " << h5_filename
                   << ". Choose a different subfile or H5 file.");
  }

  const auto functions_of_time = deserialize<std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>>(
      serialized_functions_of_time->data());

  if (not functions_of_time.contains(function_of_time_name)) {
    ERROR_NO_TRACE("No function of time named "
                   << function_of_time_name << " in the subfile "
                   << subfile_name << " of the H5 file " << h5_filename);
  }

  const auto& function_of_time = functions_of_time.at(function_of_time_name);
  const std::array<double, 2> time_bounds = function_of_time->time_bounds();

  if (time.has_value() and
      (time.value() < time_bounds[0] or time.value() > time_bounds[1])) {
    using ::operator<<;
    ERROR_NO_TRACE(function_of_time_name
                   << ": The requested time " << time
                   << " is out of the range of the function of time "
                   << time_bounds << " from the subfile " << subfile_name
                   << " of the H5 file " << h5_filename);
  }

  return function_of_time->get_clone();
}
}  // namespace

FromVolumeFile::FromVolumeFile(std::string h5_filename,
                               std::string subfile_name, const bool replay)
    : h5_filename_(std::move(h5_filename)),
      subfile_name_(std::move(subfile_name)),
      replay_(replay) {}

FunctionsOfTimeMap FromVolumeFile::retrieve_function_of_time(
    const std::unordered_set<std::string>& function_of_time_names,
    const std::optional<double>& time) const {
  FunctionsOfTimeMap result{};
  for (const std::string& name : function_of_time_names) {
    result[name] =
        get_function_of_time(name, h5_filename_, subfile_name_, time);
  }

  return result;
}
}  // namespace domain::creators::time_dependent_options
