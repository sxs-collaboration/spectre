// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/IO/Observers/MockH5.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/StdHelpers.hpp"

namespace TestHelpers::observers {
void MockDat::append(const std::vector<std::string>& legend,
                     const std::vector<double>& new_data) {
  auto& matrix = data_.second;
  // If this isn't the first call, do some checks. Otherwise, assign the
  // legend. Use `if` with ERROR rather than ASSERT so these bugs can be caught
  // in Release mode
  if (matrix.rows() != 0) {
    if (matrix.columns() != new_data.size()) {
      ERROR(
          "Size of supplied data does not match number of columns in the "
          "existing matrix. Data size: "
          << new_data.size() << ", Matrix columns: " << matrix.columns());
    }
    if (data_.first != legend) {
      ERROR(
          "Supplied legend is not the same as the existing legend. Supplied "
          "legend: "
          << legend << ", Existing legend: " << data_.first);
    }
  } else {
    data_.first = legend;
  }

  matrix.resize(matrix.rows() + 1, new_data.size(), true);
  for (size_t i = 0; i < new_data.size(); i++) {
    matrix(matrix.rows() - 1, i) = new_data[i];
  }
}

void MockDat::check_data(const std::string& function_name) const {
  // Use `if` with ERROR rather than ASSERT so these bugs can be caught in
  // Release mode
  if (data_.first.empty()) {
    ERROR("Cannot get " << function_name << ". Append some data first.");
  }
}

MockDat& MockH5File::get_dat(const std::string& subfile_path) {
  check_subfile(subfile_path);
  return subfiles_.at(subfile_path);
}

const MockDat& MockH5File::get_dat(const std::string& subfile_path) const {
  check_subfile(subfile_path);
  return subfiles_.at(subfile_path);
}

MockDat& MockH5File::try_insert(const std::string& subfile_path) {
  subfiles_.try_emplace(subfile_path);

  return subfiles_.at(subfile_path);
}

void MockH5File::check_subfile(const std::string& subfile_path) const {
  // Use `if` with ERROR rather than ASSERT so these bugs can be caught in
  // Release mode
  if (subfiles_.count(subfile_path) != 1) {
    ERROR("Cannot get " << subfile_path
                        << " from MockH5File. Path does not exist.");
  }
}
}  // namespace TestHelpers::observers
