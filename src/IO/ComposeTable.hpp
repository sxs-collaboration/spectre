// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace io {
/*!
 * \brief Reader for the CompOSE (https://compose.obspm.fr/home) ASCII tabulated
 * equations of state.
 *
 * The directory from which the data is read must contain the `eos.quantities`,
 * `eos.parameters`, and `eos.table`.
 */
class ComposeTable {
 public:
  ComposeTable() = default;
  explicit ComposeTable(std::string directory_to_read_from);

  const std::unordered_map<std::string, DataVector>& data() const {
    return data_;
  }

  const DataVector& data(const std::string& quantity_name) const {
    return data_.at(quantity_name);
  }

  const std::vector<std::string>& available_quantities() const {
    return available_quantities_;
  }

  const std::array<double, 2>& number_density_bounds() const {
    return number_density_bounds_;
  }

  const std::array<double, 2>& temperature_bounds() const {
    return temperature_bounds_;
  }

  const std::array<double, 2>& electron_fraction_bounds() const {
    return electron_fraction_bounds_;
  }

  size_t number_density_number_of_points() const {
    return number_density_number_of_points_;
  }

  size_t temperature_number_of_points() const {
    return temperature_number_of_points_;
  }

  size_t electron_fraction_number_of_points() const {
    return electron_fraction_number_of_points_;
  }

  bool beta_equilibrium() const { return beta_equilibrium_; }

  bool number_density_log_spacing() const {
    return number_density_log_spacing_;
  }

  bool temperature_log_spacing() const { return temperature_log_spacing_; }

  bool electron_fraction_log_spacing() const {
    return electron_fraction_log_spacing_;
  }

  void pup(PUP::er& p);

 private:
  void parse_eos_quantities();
  void parse_eos_parameters();
  void parse_eos_table();

  static const std::vector<std::string>
      compose_regular_and_additional_index_to_names_;
  static const std::vector<std::string> compose_derivative_index_to_names_;
  std::string directory_to_read_from_;
  std::vector<std::string> available_quantities_;
  std::array<size_t, 3> interpolation_order_;
  std::array<double, 2> number_density_bounds_;
  std::array<double, 2> temperature_bounds_;
  std::array<double, 2> electron_fraction_bounds_;
  size_t number_density_number_of_points_;
  size_t temperature_number_of_points_;
  size_t electron_fraction_number_of_points_;
  size_t table_size_;
  bool beta_equilibrium_;
  bool number_density_log_spacing_;
  bool temperature_log_spacing_;
  bool electron_fraction_log_spacing_;
  std::unordered_map<std::string, DataVector> data_;
};
}  // namespace io
