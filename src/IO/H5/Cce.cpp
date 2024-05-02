// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Cce.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <hdf5.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/OpenGroup.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
Cce::Cce(const bool exists, detail::OpenGroup&& group, const hid_t /*location*/,
         const std::string& name, const size_t l_max, const uint32_t version)
    : group_(std::move(group)),
      name_(extension() == name.substr(name.size() > extension().size()
                                           ? name.size() - extension().size()
                                           : 0)
                ? name
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      l_max_(l_max),
      legend_([this]() {
        std::vector<std::string> legend;
        legend.reserve(2 * square(l_max_ + 1) + 1);
        legend.emplace_back("time");
        for (int i = 0; i <= static_cast<int>(l_max_); ++i) {
          for (int j = -i; j <= i; ++j) {
            legend.push_back(MakeString{} << "Real Y_" << i << "," << j);
            legend.push_back(MakeString{} << "Imag Y_" << i << "," << j);
          }
        }
        return legend;
      }()),
      cce_group_(group_.id(), name_, h5::AccessType::ReadWrite) {
  for (const std::string& bondi_var : bondi_variables_) {
    // We will set the id below
    bondi_datasets_[bondi_var] = DataSet{-1, {0, legend_.size()}};
  }

  if (exists) {
    {
      // We treat this as an internal version for now. We'll need to deal with
      // proper versioning later.

      // Check if the version exists before calling the open_version.
      const htri_t version_exists = H5Aexists(cce_group_.id(), "version.ver");
      if (version_exists != 0) {
        const Version open_version(true, detail::OpenGroup{}, cce_group_.id(),
                                   "version");
        version_ = open_version.get_version();
      }
    }
    {
      const htri_t header_exists = H5Aexists(cce_group_.id(), "header.hdr");
      if (header_exists != 0) {
        const Header header(true, detail::OpenGroup{}, cce_group_.id(),
                            "header");
        header_ = header.get_header();
      }
    }

    for (const std::string& bondi_var : bondi_variables_) {
      DataSet& dataset = bondi_datasets_.at(bondi_var);
      dataset.id =
          H5Dopen2(cce_group_.id(), bondi_var.c_str(), h5::h5p_default());
      CHECK_H5(dataset.id, "Failed to open dataset");

      hid_t space_id = H5Dget_space(dataset.id);
      std::array<hsize_t, 2> max_dims{};
      if (2 != H5Sget_simple_extent_dims(space_id, dataset.size.data(),
                                         max_dims.data())) {
        ERROR("Invalid number of dimensions in cce file " << bondi_var
                                                          << " on disk.");
      }
      CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");

      if (legend_ != read_rank1_attribute<std::string>(dataset.id, "Legend"s)) {
        ERROR("l_max from cce file " << bondi_var
                                     << " does not match l_max in constructor");
      }
    }
  } else {  // file does not exist
    {
      Version open_version(false, detail::OpenGroup{}, cce_group_.id(),
                           "version", version_);
    }
    {
      Header header(false, detail::OpenGroup{}, cce_group_.id(), "header");
      header_ = header.get_header();
    }
    // So this file is compatible with the sxs/scri python packages offered by
    // the SXS collaboration
    write_to_attribute(cce_group_.id(), sxs_format_str_, sxs_version_str_);

    for (const std::string& bondi_var : bondi_variables_) {
      DataSet& dataset = bondi_datasets_.at(bondi_var);
      dataset.id = h5::detail::create_extensible_dataset(
          cce_group_.id(), bondi_var, dataset.size,
          std::array<hsize_t, 2>{{4, legend_.size()}},
          {{h5s_unlimited(), legend_.size()}});
      CHECK_H5(dataset.id, "Failed to create dataset");

      write_to_attribute(dataset.id, "Legend"s, legend_);
      // So this dataset is compatible with the sxs/scri python packages offered
      // by the SXS collaboration
      write_to_attribute(dataset.id, sxs_format_str_, sxs_version_str_);
    }
  }
}

Cce::~Cce() {
#if defined(__GNUC__) and not defined(__clang__)
// The pragma here is used to suppress the warning that the compile time branch
// of `ERROR` will always call `terminate()` because it throws an error.
// Throwing that error is a code path that will never actually be entered at
// runtime, so we suppress the warning here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
#endif
  // We don't use structured bindings because there was a bug that was fixed in
  // C++20 about capturing structured bindings in a lambda (the lambda is in the
  // internals of CHECK_H5), so older compilers that we support may not have
  // fixed this bug.
  for (const auto& name_and_dataset : bondi_datasets_) {
    const auto& name = name_and_dataset.first;
    const auto& dataset = name_and_dataset.second;
    CHECK_H5(H5Dclose(dataset.id), "Failed to close dataset " << name);
  }
#if defined(__GNUC__) and not defined(__clang__)
#pragma GCC diagnostic pop
#endif
}

void Cce::append(
    const std::unordered_map<std::string, std::vector<double>>& data) {
  for (const std::string& bondi_var : bondi_variables_) {
    if (not data.contains(bondi_var)) {
      ERROR("Passed in data does not contain the bondi variable " << bondi_var);
    }
    DataSet& dataset = bondi_datasets_.at(bondi_var);

    const std::vector<double>& vec_data = data.at(bondi_var);
    if (vec_data.size() != dataset.size[1]) {
      ERROR("Cannot add columns to Cce files. Current number of columns is "
            << dataset.size[1] << " but received " << vec_data.size()
            << " entries.");
    }

    dataset.size =
        h5::append_to_dataset(dataset.id, bondi_var, vec_data, 1, dataset.size);
  }
}

std::unordered_map<std::string, Matrix> Cce::get_data() const {
  std::unordered_map<std::string, Matrix> result{};

  for (const std::string& bondi_var : bondi_variables_) {
    const DataSet& dataset = bondi_datasets_.at(bondi_var);
    result[bondi_var] = h5::retrieve_dataset(dataset.id, dataset.size);
  }

  return result;
}

Matrix Cce::get_data(const std::string& bondi_variable_name) const {
  check_bondi_variable(bondi_variable_name);

  const DataSet& dataset = bondi_datasets_.at(bondi_variable_name);
  return h5::retrieve_dataset(dataset.id, dataset.size);
}

std::unordered_map<std::string, Matrix> Cce::get_data_subset(
    const std::vector<size_t>& these_ell, const size_t first_row,
    const size_t num_rows) const {
  std::unordered_map<std::string, Matrix> result{};

  for (const std::string& bondi_var : bondi_variables_) {
    result[bondi_var] =
        get_data_subset(bondi_var, these_ell, first_row, num_rows);
  }

  return result;
}

Matrix Cce::get_data_subset(const std::string& bondi_variable_name,
                            const std::vector<size_t>& these_ell,
                            const size_t first_row,
                            const size_t num_rows) const {
  check_bondi_variable(bondi_variable_name);

  if (alg::any_of(these_ell,
                  [this](const size_t ell) { return ell > l_max_; })) {
    ERROR("One (or more) of the requested ells "
          << these_ell << " is larger than the l_max " << l_max_);
  }

  if (these_ell.empty()) {
    return {num_rows, 0, 0.0};
  }

  // Always grab the time
  std::vector<size_t> these_columns{0};

  // For a given ell, we nedd the first column index and the last column index.
  // For a given ell, the total number of coefs is given by
  // f(l) = 2 * (l + 1)^2 (first factor of 2 comes from having both real and
  // imag components). Therefore, the first column index is just f(ell - 1) + 1,
  // and the last column index is just f(ell). The +1 is because we have a
  // time column
  for (size_t ell : these_ell) {
    if (ell == 0_st) {
      these_columns.emplace_back(1);
      these_columns.emplace_back(2);
    } else {
      const size_t first_col = 2 * square(ell) + 1;
      const size_t last_col = 2 * square(ell + 1);
      for (size_t col = first_col; col <= last_col; col++) {
        these_columns.emplace_back(col);
      }
    }
  }

  if (num_rows == 0) {
    return {0, these_columns.size(), 0.0};
  }

  const DataSet& dataset = bondi_datasets_.at(bondi_variable_name);
  return h5::retrieve_dataset_subset(dataset.id, these_columns, first_row,
                                     num_rows, dataset.size);
}

void Cce::check_bondi_variable(const std::string& bondi_variable_name) const {
  if (not bondi_variables_.contains(bondi_variable_name)) {
    ERROR("Requested bondi variable " << bondi_variable_name
                                      << " not available");
  }
}

const std::array<hsize_t, 2>& Cce::get_dimensions(
    const std::string& bondi_variable_name) const {
  check_bondi_variable(bondi_variable_name);

  return bondi_datasets_.at(bondi_variable_name).size;
}
}  // namespace h5
