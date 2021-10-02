// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/StellarCollapseEos.hpp"

#include <algorithm>
#include <hdf5.h>
#include <ostream>

#include "IO/H5/AccessType.hpp"
#include "IO/H5/Helpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace h5 {
StellarCollapseEos::StellarCollapseEos(bool exists, detail::OpenGroup&& group,
                                       const hid_t /*location*/,
                                       const std::string& name)
    : group_(std::move(group)) {
  if (not exists) {
    ERROR("The subfile '" << name << "' does not exist");
  }
  if (name != "/") {
    root_group_ = std::move(group_);
    group_ = detail::OpenGroup(root_group_.id(), name, AccessType::ReadOnly);
  }
}

template <typename T>
T StellarCollapseEos::get_scalar_dataset(
    const std::string& dataset_name) const {
  return read_data<0, T>(group_.id(), dataset_name);
}

std::vector<double> StellarCollapseEos::get_rank1_dataset(
    const std::string& dataset_name) const {
  return read_data<1, std::vector<double>>(group_.id(), dataset_name);
}

boost::multi_array<double, 3> StellarCollapseEos::get_rank3_dataset(
    const std::string& dataset_name) const {
  return read_data<3, boost::multi_array<double, 3>>(group_.id(), dataset_name);
}

template double StellarCollapseEos::get_scalar_dataset(
    const std::string& dataset_name) const;

template int StellarCollapseEos::get_scalar_dataset(
    const std::string& dataset_name) const;

}  // namespace h5
