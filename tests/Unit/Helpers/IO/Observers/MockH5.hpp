// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"

namespace TestHelpers::observers {
/*!
 * \brief Class meant to mock h5::Dat in the testing framework.
 *
 * Currently, this class is only used inside a MockH5 object. The methods of
 * this class are similar to that of h5::Dat to keep a familiar interface, but
 * they may not be identical.
 */
struct MockDat {
  void append(const std::vector<std::string>& legend,
              const std::vector<double>& new_data);

  const Matrix& get_data() const {
    check_data("data");
    return data_.second;
  }

  const std::vector<std::string>& get_legend() const {
    check_data("legend");
    return data_.first;
  }

  void pup(PUP::er& p) { p | data_; }  // NOLINT

 private:
  void check_data(const std::string& function_name) const;
  std::pair<std::vector<std::string>, Matrix> data_{};
};

/*!
 * \brief Class meant to mock h5::H5File in the testing framework.
 *
 * The methods of this class are similar to that of h5::H5File to keep a
 * familiar interface, but they may not be identical.
 */
struct MockH5File {
  MockDat& get_dat(const std::string& subfile_path);

  const MockDat& get_dat(const std::string& subfile_path) const;

  MockDat& try_insert(const std::string& subfile_path);

  void pup(PUP::er& p) { p | subfiles_; }  // NOLINT

 private:
  void check_subfile(const std::string& subfile_path) const;

  std::unordered_map<std::string, MockDat> subfiles_{};
};
}  // namespace TestHelpers::observers
