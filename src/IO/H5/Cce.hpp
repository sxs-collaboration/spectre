// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::Cce

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

/// \cond
class Matrix;
/// \endcond

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Represents Cauchy-Characteristic Extraction (CCE) bondi variables
 * inside of an HDF5 file.
 *
 * Within a Cce object, there are several datasets that correspond to the
 * `Cce::scri_plus_interpolation_set` bondi variables at future null infinity
 * represented as complex spherical harmonic coefficients. The names of these
 * bondi variables are
 *
 * - `EthInertialRetardedTime`
 * - `News`
 * - `Psi0`
 * - `Psi1`
 * - `Psi2`
 * - `Psi3`
 * - `Psi4`
 * - `Strain`
 *
 * Each dataset has a couple H5 attributes:
 *
 * - `Legend` which is determined by the `l_max` massed to the constructor.
 * - `sxs_format` which is a string. The value is "SpECTRE_CCE_v1"
 *
 * The Cce object itself also has the `sxs_format` attribute with the same
 * value, along with a `version` and `header` attribute.
 *
 * The columns of data are stored in each dataset in the following order:
 *
 * - `time`
 * - `Real Y_0,0`
 * - `Imag Y_0,0`
 * - `Real Y_1,-1`
 * - `Imag Y_1,-1`
 * - `Real Y_1,0`
 * - `Imag Y_1,0`
 * - `Real Y_1,1`
 * - `Imag Y_1,1`
 * - ...
 *
 * and so on until you reach the coefficients for `l_max`.
 *
 * \note This class does not do any caching of data so all data is written as
 * soon as append() is called.
 */
class Cce : public h5::Object {
  struct DataSet {
    hid_t id;
    std::array<hsize_t, 2> size;
  };

 public:
  /// \cond HIDDEN_SYMBOLS
  static std::string extension() { return ".cce"; }

  Cce(bool exists, detail::OpenGroup&& group, hid_t location,
      const std::string& name, size_t l_max, uint32_t version = 1);

  Cce(const Cce& /*rhs*/) = delete;
  Cce& operator=(const Cce& /*rhs*/) = delete;
  Cce(Cce&& /*rhs*/) = delete;             // NOLINT
  Cce& operator=(Cce&& /*rhs*/) = delete;  // NOLINT

  ~Cce() override;
  /// \endcond HIDDEN_SYMBOLS

  /*!
   * \brief For each bondi variable name, appends the \p data to the dataset in
   * the H5 file.
   *
   * \details The `data.at(name).size()` must be the same as the number of
   * columns already in the dataset. Also, all bondi variable names listed in
   * the constructor of `h5::Cce` must be present in the \p data.
   */
  void append(const std::unordered_map<std::string, std::vector<double>>& data);

  /// @{
  /*!
   * \brief Return all currently stored data in the `h5::Cce` file in the form
   * of a `Matrix` for each bondi variable
   */
  std::unordered_map<std::string, Matrix> get_data() const;

  Matrix get_data(const std::string& bondi_variable_name) const;
  /// @}

  /// @{
  /*!
   * \brief Get only some values of $\ell$ over a range of rows
   *
   * \details The `time` column is always returned. The coefficients will be
   * returned in the order that you requested them in. No sorting is done
   * internally. All requested \p these_ell must be less that or equal to the \p
   * l_max that this file was constructed with. Also both the first and last row
   * requested must be less than or equal to the total number of rows.
   */
  std::unordered_map<std::string, Matrix> get_data_subset(
      const std::vector<size_t>& these_ell, size_t first_row = 0,
      size_t num_rows = 1) const;

  Matrix get_data_subset(const std::string& bondi_variable_name,
                         const std::vector<size_t>& these_ell,
                         size_t first_row = 0, size_t num_rows = 1) const;
  /// @}

  /*!
   * \brief Return the legend. All bondi variables have the same legend.
   */
  const std::vector<std::string> get_legend() const { return legend_; }

  /*!
   * \brief Return the number of rows (first index) and columns (second index)
   * of the \p bondi_variable_name dataset. All bondi variables will have the
   * same dimensions.
   */
  const std::array<hsize_t, 2>& get_dimensions(
      const std::string& bondi_variable_name) const;

  /*!
   * \brief The header of the Cce file
   */
  const std::string& get_header() const { return header_; }

  /*!
   * \brief The user-specified version number of the Cce file
   *
   * \note h5::Version returns a uint32_t, so we return one here too for the
   * version
   */
  uint32_t get_version() const { return version_; }

  /*!
   * \brief Path to this Cce file.
   */
  const std::string& subfile_path() const override { return path_; }

 private:
  void check_bondi_variable(const std::string& bondi_variable_name) const;
  /// \cond HIDDEN_SYMBOLS
  detail::OpenGroup group_;
  std::string name_;
  std::string path_;
  uint32_t version_;
  size_t l_max_;
  std::vector<std::string> legend_{};
  detail::OpenGroup cce_group_{};
  std::string header_;
  std::unordered_map<std::string, DataSet> bondi_datasets_;
  std::unordered_set<std::string> bondi_variables_{"EthInertialRetardedTime",
                                                   "News",
                                                   "Psi0",
                                                   "Psi1",
                                                   "Psi2",
                                                   "Psi3",
                                                   "Psi4",
                                                   "Strain"};
  std::string sxs_format_str_{"sxs_format"};
  std::string sxs_version_str_{"SpECTRE_CCE_v1"};
  /// \endcond HIDDEN_SYMBOLS
};
}  // namespace h5
