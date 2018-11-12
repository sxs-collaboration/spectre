// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class h5::StellarCollapseEos

#pragma once

#include <hdf5.h>
#include <string>
#include <vector>

#include "DataStructures/BoostMultiArray.hpp" // IWYU pragma: keep
#include "IO/H5/Object.hpp"
#include "IO/H5/OpenGroup.hpp"

// IWYU pragma: no_include <boost/multi_array.hpp>

namespace h5 {
/*!
 * \ingroup HDF5Group
 * \brief Reads in tabulated equation of state file from
 * [stellarcollapse.org](https://stellarcollapse.org)
 *
 * Reads in H5 file containing data for tabulated equation of state.
 * Contains functions to obtain thermodynamic quantities from the file,
 * stored as either rank-1 or rank-3 datasets.
 *
 * It is assumed that the file is in the format of the
 * [SRO (Schneider, Roberts, Ott 2017) Equation of State files](https://
 * stellarcollapse.org/SROEOS)
 *
 * The description of each dataset in the file can be found
 * [here](https://bitbucket.org/andschn/sroeos/src/master/
 * User_Guide/User_Guide.pdf?fileviewer=file-view-default)
 *
 */
class StellarCollapseEos : public h5::Object {
 public:
  /// \cond
  // The root-level HDF5 group in the SRO Equation of State files does not
  // have an extension in its group name
  static std::string extension() noexcept { return ""; }

  StellarCollapseEos(bool exists, detail::OpenGroup&& group, hid_t location,
                     const std::string& /*name*/) noexcept;

  StellarCollapseEos(const StellarCollapseEos& /*rhs*/) = delete;
  StellarCollapseEos& operator=(const StellarCollapseEos& /*rhs*/) = delete;
  StellarCollapseEos(StellarCollapseEos&& /*rhs*/) noexcept = delete;
  StellarCollapseEos& operator=(StellarCollapseEos&& /*rhs*/) noexcept = delete;

  ~StellarCollapseEos() override = default;
  /// \endcond

  /*!
   * \ingroup HDF5Group
   * \brief reads a rank-0 dataset (contains only one element)
   */
  template <typename T>
  T get_scalar_dataset(const std::string& dataset_name) const noexcept;

  /*!
   * \ingroup HDF5Group
   * \brief reads a dataset with elements along 1 dimension
   */
  std::vector<double> get_rank1_dataset(const std::string& dataset_name) const
      noexcept;

  /*!
   * \ingroup HDF5Group
   * \brief reads a dataset with elements along 3 dimensions
   */
  boost::multi_array<double, 3> get_rank3_dataset(
      const std::string& dataset_name) const noexcept;

 private:
  detail::OpenGroup root_group_;
  detail::OpenGroup group_;
};
}  // namespace h5
