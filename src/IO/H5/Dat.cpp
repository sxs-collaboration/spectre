// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Dat.hpp"

#include <algorithm>
#include <hdf5.h>
#include <iosfwd>
#include <memory>
#include <ostream>

#include "DataStructures/Matrix.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

namespace h5 {
Dat::Dat(const bool exists, detail::OpenGroup&& group, const hid_t location,
         const std::string& name, std::vector<std::string> legend,
         const uint32_t version)
    : group_(std::move(group)),
      name_(extension() == name.substr(name.size() > extension().size()
                                           ? name.size() - extension().size()
                                           : 0)
                ? name
                : name + extension()),
      path_(group_.group_path_with_trailing_slash() + name),
      version_(version),
      legend_(std::move(legend)),
      size_{{0, legend_.size()}} {
  if (exists) {
    dataset_id_ = H5Dopen2(location, name_.c_str(), h5::h5p_default());
    CHECK_H5(dataset_id_, "Failed to open dataset");
    {
      // We treat this as an internal version for now. We'll need to deal with
      // proper versioning later.

      // Check if the version exists before calling the open_version.
      const htri_t version_exists = H5Aexists(dataset_id_, "version.ver");
      if (version_exists != 0) {
        const Version open_version(true, detail::OpenGroup{}, dataset_id_,
                                   "version");
        version_ = open_version.get_version();
      }
    }
    {
      const htri_t header_exists = H5Aexists(dataset_id_, "header.hdr");
      if (header_exists != 0) {
        const Header header(true, detail::OpenGroup{}, dataset_id_, "header");
        header_ = header.get_header();
      }
    }

    hid_t space_id = H5Dget_space(dataset_id_);
    std::array<hsize_t, 2> maxdims{};
    if (2 !=
        H5Sget_simple_extent_dims(space_id, size_.data(), maxdims.data())) {
      ERROR("Invalid number of dimensions in file on disk.");  // LCOV_EXCL_LINE
    }
    CHECK_H5(H5Sclose(space_id), "Failed to close dataspace");
    legend_ = read_rank1_attribute<std::string>(dataset_id_, "Legend"s);
    size_[1] = legend_.size();
  } else {  // file does not exist
    dataset_id_ = h5::detail::create_extensible_dataset(
        location, name_, size_, std::array<hsize_t, 2>{{4, legend_.size()}},
        {{h5s_unlimited(), legend_.size()}});
    CHECK_H5(dataset_id_, "Failed to create dataset");

    {
      Version open_version(false, detail::OpenGroup{}, dataset_id_, "version",
                           version_);
    }
    {
      Header header(false, detail::OpenGroup{}, dataset_id_, "header");
      header_ = header.get_header();
    }
    // Capitalized for compatibility with SpEC output
    write_to_attribute(dataset_id_, "Legend"s, legend_);
  }
}

Dat::~Dat() {
#ifdef __clang__
  CHECK_H5(H5Dclose(dataset_id_), "Failed to close dataset");
#else
// The pragma here is used to suppress the warning that the compile time branch
// of `ERROR` will always call `terminate()` because it throws an error.
// Throwing that error is a code path that will never actually be entered at
// runtime, so we suppress the warning here.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wterminate"
  CHECK_H5(H5Dclose(dataset_id_), "Failed to close dataset");
#pragma GCC diagnostic pop
#endif
}

void Dat::append(const std::vector<double>& data) {
  if (data.size() != size_[1]) {
    ERROR("Cannot add columns to Dat files. Current number of columns is "
          << size_[1] << " but received " << data.size() << " entries.");
  }

  size_ = h5::append_to_dataset(dataset_id_, name_, data, 1, size_);
}

void Dat::append(const std::vector<std::vector<double>>& data) {
  if (data.empty()) {
    return;
  }
  if (data[0].size() != size_[1]) {
    ERROR("Cannot add columns to Dat files. Current number of columns is "
          << size_[1] << " but received " << data[0].size() << " entries.");
  }
  const std::vector<double> contiguous_data =
      [](const std::vector<std::vector<double>>& ldata) {
        std::vector<double> result(ldata.size() * ldata[0].size());
        for (size_t i = 0; i < ldata.size(); ++i) {
          if (ldata[i].size() != ldata[0].size()) {
            ERROR(
                "Each member of the vector<vector<double>> must be of the same "
                "size, ie the number of columns must be the same.");
          }
          result.insert(
              result.begin() + static_cast<std::ptrdiff_t>(i * ldata[i].size()),
              ldata[i].begin(), ldata[i].end());
        }
        return result;
      }(data);

  size_ = h5::append_to_dataset(dataset_id_, name_, contiguous_data,
                                data.size(), size_);
}

void Dat::append(const Matrix& data) {
  if (0 == data.rows() * data.columns()) {
    return;
  }
  if (data.columns() != size_[1]) {
    ERROR("Cannot add columns to Dat files. Current number of columns is "
          << size_[1] << " but received " << data.columns() << " entries.");
  }
  // can't use begin() and end() because the ordering ends up wrong
  const std::vector<double> contiguous_data = [](const Matrix& ldata) {
    std::vector<double> result(ldata.rows() * ldata.columns());
    for (size_t i = 0; i < ldata.rows(); ++i) {
      for (size_t j = 0; j < ldata.columns(); ++j) {
        result[j + i * ldata.columns()] = ldata(i, j);
      }
    }
    return result;
  }(data);

  size_ = h5::append_to_dataset(dataset_id_, name_, contiguous_data,
                                data.rows(), size_);
}

Matrix Dat::get_data() const {
  return h5::retrieve_dataset(dataset_id_, size_);
}

Matrix Dat::get_data_subset(const std::vector<size_t>& these_columns,
                            const size_t first_row,
                            const size_t num_rows) const {
  return retrieve_dataset_subset(dataset_id_, these_columns, first_row,
                                 num_rows, size_);
}
}  // namespace h5
