// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

/// \cond
class DataVector;
/// \endcond

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief An equation of state table subfile written inside an H5 file.
 *
 * ### Data Layout
 *
 * To be consistent with the CompOSE ASCII table format, the data is stored in a
 * last independent variable varies fastest. For example, if you had independent
 * variables \f$\rho\f$, \f$T\f$, and \f$Y_e\f$, then \f$Y_e\f$ varies fastest
 * while \f$\rho\f$ varies slowest.
 */
class EosTable : public h5::Object {
 public:
  static std::string extension() { return ".eos"; }

  /// Constructor used when writing a new equation of state.
  EosTable(bool subfile_exists, detail::OpenGroup&& group, hid_t location,
           const std::string& name,
           std::vector<std::string> independent_variable_names,
           std::vector<std::array<double, 2>> independent_variable_bounds,
           std::vector<size_t> independent_variable_number_of_points,
           std::vector<bool> independent_variable_uses_log_spacing,
           bool beta_equilibrium, uint32_t version = 1);

  /// Constructor used when reading in an equation of state.
  EosTable(bool subfile_exists, detail::OpenGroup&& group, hid_t location,
           const std::string& name);

  EosTable(const EosTable& /*rhs*/) = delete;
  EosTable& operator=(const EosTable& /*rhs*/) = delete;
  EosTable(EosTable&& /*rhs*/) = delete;             // NOLINT
  EosTable& operator=(EosTable&& /*rhs*/) = delete;  // NOLINT

  ~EosTable() override = default;

  /*!
   * \returns the header of the EosTable file
   */
  const std::string& get_header() const { return header_; }

  /*!
   * \returns the user-specified version number of the EosTable file
   *
   * \note h5::Version returns a uint32_t, so we return one here too for the
   * version
   */
  uint32_t get_version() const { return version_; }

  const std::string& subfile_path() const override { return path_; }

  /*!
   * \brief Write a thermodynamic quantity to disk.
   */
  void write_quantity(std::string name, const DataVector& data);

  /*!
   * \brief Read a thermodynamic quantity to disk.
   */
  DataVector read_quantity(const std::string& name) const;

  /// The available thermodynamic quantities.
  const std::vector<std::string> available_quantities() const {
    return available_quantities_;
  }

  /// Number of independent variables.
  size_t number_of_independent_variables() const {
    return independent_variable_names_.size();
  }

  /// Names of the independent variables.
  const std::vector<std::string>& independent_variable_names() const {
    return independent_variable_names_;
  }

  /// Lower and upper bounds of the independent variables.
  const std::vector<std::array<double, 2>>& independent_variable_bounds()
      const {
    return independent_variable_bounds_;
  }

  /// The number of points for each of the independent variables.
  const std::vector<size_t>& independent_variable_number_of_points() const {
    return independent_variable_number_of_points_;
  }

  /// Whether each independent variable is in log spacing. Linear spacing is
  /// used if `false`.
  const std::vector<bool>& independent_variable_uses_log_spacing() const {
    return independent_variable_uses_log_spacing_;
  }

  /// `true` if the EOS is in beta equilibrium.
  bool beta_equilibrium() const { return beta_equilibrium_; }

 private:
  detail::OpenGroup group_{};
  std::string name_{};
  std::string path_{};
  uint32_t version_{};
  detail::OpenGroup eos_table_group_{};
  std::string header_{};

  std::vector<std::string> independent_variable_names_{};
  std::vector<std::array<double, 2>> independent_variable_bounds_{};
  std::vector<size_t> independent_variable_number_of_points_{};
  std::vector<bool> independent_variable_uses_log_spacing_{};
  bool beta_equilibrium_ = false;
  std::vector<std::string> available_quantities_{};
};
}  // namespace h5
