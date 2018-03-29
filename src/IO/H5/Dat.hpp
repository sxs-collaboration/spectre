// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::Dat

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <vector>

#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

/// \cond
class Matrix;
/// \endcond

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Represents a multicolumn dat file inside an HDF5 file
 *
 * A Dat object represents a dat file inside an H5File. A dat file is a
 * multicolumn text file with a header describing what each column represents.
 * Typically dat files are space or tab delimited, and often represent time
 * series data. One common use for them is writing out error norms over the
 * computational domain as a function of time. Inside the H5File they are stored
 * as a string header, and a matrix of doubles holding the data. One problem
 * encountered with dat files is that they quickly increase the file count
 * causing users to run into number of file limitations on HPC systems. Since
 * multiple Dat objects can be stored inside a single H5File the problem of many
 * different dat files being stored as individual files is solved.
 *
 * \note This class does not do any caching of data so all data is written as
 * soon as append() is called.
 */
class Dat : public h5::Object {
 public:
  /// \cond HIDDEN_SYMBOLS
  static std::string extension() { return ".dat"; }

  Dat(bool exists, detail::OpenGroup&& group, hid_t location,
      const std::string& name, std::vector<std::string> legend = {},
      uint32_t version = 1);

  Dat(const Dat& /*rhs*/) = delete;
  Dat& operator=(const Dat& /*rhs*/) = delete;
  Dat(Dat&& /*rhs*/) noexcept = delete;             // NOLINT
  Dat& operator=(Dat&& /*rhs*/) noexcept = delete;  // NOLINT

  ~Dat() override;
  /// \endcond HIDDEN_SYMBOLS

  /*!
   * \requires `data.size()` is the same as the number of columns in the file
   * \effects appends `data` to the Dat file
   */
  void append(const std::vector<double>& data);

  /*!
   * \requires `data[0].size()` is the same as the number of columns in the file
   * \effects appends `data` to the Dat file
   */
  void append(const std::vector<std::vector<double>>& data);

  /*!
   * \requires `data.columns()` is the same as the number of columns in the file
   * \effects appends `data` to the Dat file
   */
  void append(const Matrix& data);

  /*!
   * \returns the legend of the Dat file
   */
  const std::vector<std::string>& get_legend() const noexcept {
    return legend_;
  }

  /*!
   * \returns all data stored in the Dat file
   *
   * \example
   * \snippet Test_H5.cpp h5dat_get_data
   */
  Matrix get_data() const;

  /*!
   * \brief Get only some columns over a range of rows
   * \requires all members of `these_columns` have a value less than the number
   * of columns, `first_row < last_row` and `last_row` is less than or equal to
   * the number of rows
   * \returns a subset of the data from the Dat file
   *
   * \example
   * \snippet Test_H5.cpp h5dat_get_subset
   */
  Matrix get_data_subset(const std::vector<size_t>& these_columns,
                         size_t first_row = 0, size_t num_rows = 1) const;

  /*!
   * \returns the number of rows (first index) and columns (second index)
   */
  const std::array<hsize_t, 2>& get_dimensions() const noexcept {
    return size_;
  }

  /*!
   * \returns the header of the Dat file
   */
  const std::string& get_header() const noexcept { return header_; }

  /*!
   * \returns the user-specified version number of the Dat file
   *
   * \note h5::Version returns a uint32_t, so we return one here too for the
   * version
   */
  uint32_t get_version() const noexcept { return version_; }

 private:
  void append_impl(hsize_t number_of_rows, const std::vector<double>& data);

  /// \cond HIDDEN_SYMBOLS
  detail::OpenGroup group_;
  std::string name_;
  uint32_t version_;
  std::vector<std::string> legend_;
  std::array<hsize_t, 2> size_;
  std::string header_;
  hid_t dataset_id_{-1};
  /// \endcond HIDDEN_SYMBOLS
};
}  // namespace h5
