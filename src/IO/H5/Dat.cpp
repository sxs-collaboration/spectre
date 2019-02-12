// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "IO/H5/Dat.hpp"

#include <algorithm>
#include <hdf5.h>
#include <iosfwd>
#include <memory>
#include <ostream>

#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/H5/CheckH5.hpp"
#include "IO/H5/Header.hpp"
#include "IO/H5/Helpers.hpp"
#include "IO/H5/Type.hpp"
#include "IO/H5/Version.hpp"
#include "IO/H5/Wrappers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

// IWYU pragma: no_include "DataStructures/Index.hpp"

namespace {
/*!
 * Given a vector of contiguous data and an array giving the dimensions of the
 * matrix returns a `vector<vector<T>>` representing the matrix.
 */
Matrix vector_to_matrix(const std::vector<double>& raw_data,
                        const std::array<hsize_t, 2>& size) {
  Matrix temp(size[0], size[1]);
  for (size_t i = 0; i < size[0]; ++i) {
    for (size_t j = 0; j < size[1]; ++j) {
      temp(i, j) = raw_data[j + i * size[1]];
    }
  }
  return temp;
}
}  // namespace

namespace h5 {
/// \cond HIDDEN_SYMBOLS
Dat::Dat(const bool exists, detail::OpenGroup&& group, const hid_t location,
         const std::string& name, std::vector<std::string> legend,
         const uint32_t version)
    : group_(std::move(group)),
      name_(extension() == name.substr(name.size() - extension().size())
                ? name
                : name + extension()),
      version_(version),
      legend_(std::move(legend)),
      size_{{0, legend_.size()}} {
  if (exists) {
    dataset_id_ = H5Dopen2(location, name_.c_str(), h5::h5p_default());
    CHECK_H5(dataset_id_, "Failed to open dataset");
    {
      // We treat this as an internal version for now. We'll need to deal with
      // proper versioning later.
      const Version open_version(true, detail::OpenGroup{}, dataset_id_,
                                 "version");
      version_ = open_version.get_version();
    }
    {
      const Header header(true, detail::OpenGroup{}, dataset_id_, "header");
      header_ = header.get_header();
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

Dat::~Dat() { CHECK_H5(H5Dclose(dataset_id_), "Failed to close dataset"); }
/// \endcond HIDDEN_SYMBOLS

void Dat::append_impl(const hsize_t number_of_rows,
                      const std::vector<double>& data) {
  {
    std::array<hsize_t, 2> read_size{}, read_max_size{};
    const hid_t dataspace_id = H5Dget_space(dataset_id_);
    CHECK_H5(dataspace_id, "Failed to get dataspace for appending");
    if (2 != H5Sget_simple_extent_dims(dataspace_id, read_size.data(),
                                       read_max_size.data())) {
      ERROR("Incorrect rank of file on disk");  // LCOV_EXCL_LINE
    }
    CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
    if (read_size != size_) {
      using ::operator<<;
      ERROR("Mismatch in the size of the read dataset. Read "
            << read_size << " but have stored " << size_
            << ". This means that another thread or process is writing data at "
               "the same time it is being written by this process.");
    }
  }

  std::array<hsize_t, 2> new_size{{size_[0] + number_of_rows, size_[1]}};
  CHECK_H5(H5Dset_extent(dataset_id_, new_size.data()),
           "Failed to append to the file '" << name_ << "'");
  const hid_t dataspace_id = H5Dget_space(dataset_id_);
  CHECK_H5(dataspace_id, "Failed to get dataspace for appending");
  CHECK_H5(H5Sselect_all(dataspace_id),
           "Failed to select dataspace for appending");
  CHECK_H5(H5Sselect_hyperslab(dataspace_id, H5S_SELECT_NOTB,
                               std::array<hsize_t, 2>{{0, 0}}.data(), nullptr,
                               size_.data(), nullptr),
           "Failed to select the new dataspace subset where the appended "
           "data would have been written.");
  const std::array<hsize_t, 2> added_size{{number_of_rows, size_[1]}};
  const hid_t memspace_id =
      H5Screate_simple(2, added_size.data(), added_size.data());
  CHECK_H5(memspace_id, "Failed to create new simple memspace while appending");
  CHECK_H5(H5Dwrite(dataset_id_, h5_type<double>(), memspace_id, dataspace_id,
                    h5::h5p_default(), data.data()),
           "Failed to append to dataset while writing");
  CHECK_H5(H5Sclose(memspace_id), "Failed to close memspace after appending");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace after appending");
  size_ = new_size;
}

void Dat::append(const std::vector<double>& data) {
  if (data.size() != size_[1]) {
    ERROR("Cannot add columns to Dat files. Current number of columns is "
          << size_[1] << " but received " << data.size() << " entries.");
  }
  append_impl(1, data);
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
  append_impl(data.size(), contiguous_data);
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
  append_impl(data.rows(), contiguous_data);
}

Matrix Dat::get_data() const {
  const hid_t dataspace_id = H5Dget_space(dataset_id_);
  CHECK_H5(dataspace_id, "Failed to get dataspace");

  std::array<hsize_t, 2> size{}, max_size{};
  if (2 !=
      H5Sget_simple_extent_dims(dataspace_id, size.data(), max_size.data())) {
    ERROR("Incorrect dimension in get_data()");  // LCOV_EXCL_LINE
  }
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");

  if (size != size_) {
    using ::operator<<;
    ERROR("Mismatch in the size of the read dataset. Read "
          << size << " but have stored " << size_
          << ". This means that another thread or process is writing data at "
             "the same time it is being read.");
  }

  std::vector<double> temp(size[0] * size[1]);
  if (0 != size[0] * size[1]) {
    CHECK_H5(H5Dread(dataset_id_, h5_type<double>(), h5::h5s_all(),
                     h5::h5s_all(), h5::h5p_default(), temp.data()),
             "Failed to read data");
  }
  return vector_to_matrix(temp, size);
}

Matrix Dat::get_data_subset(const std::vector<size_t>& these_columns,
                            const size_t first_row,
                            const size_t num_rows) const {
  Expects(first_row + num_rows <= size_[0]);
  Expects(std::all_of(these_columns.begin(),
                      these_columns.end(), [size = size_](const auto& column) {
                        return column < size[1];
                      }));

  const auto num_cols = these_columns.size();
  if (0 == num_cols * num_rows) {
    return Matrix(num_rows, num_cols, 0.0);
  }

  const hid_t dataspace_id = H5Dget_space(dataset_id_);
  CHECK_H5(dataspace_id, "Failed to get dataspace");
  std::array<hsize_t, 2> size{}, max_size{};
  if (2 !=
      H5Sget_simple_extent_dims(dataspace_id, size.data(), max_size.data())) {
    ERROR("Incorrect dimension in get_data()");  // LCOV_EXCL_LINE
  }
  if (size != size_) {
    using ::operator<<;
    CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
    ERROR("Mismatch in the size of the read dataset. Read "
          << size << " but have stored " << size_
          << ". This means that another thread or process is writing data at "
             "the same time it is being read.");
  }

  CHECK_H5(H5Sselect_none(dataspace_id),
           "Failed to select none of the dataspace");
  for (auto& column : these_columns) {
    const std::array<hsize_t, 2> start{
        {first_row, static_cast<hsize_t>(column)}};
    // offset between blocks (have only one anyway)
    const std::array<hsize_t, 2> stride{{1, 1}};
    const std::array<hsize_t, 2> count{{1, 1}};
    const std::array<hsize_t, 2> block{{num_rows, 1}};

    CHECK_H5(H5Sselect_hyperslab(dataspace_id, H5S_SELECT_OR, start.data(),
                                 stride.data(), count.data(), block.data()),
             "Failed to select column " << column);
  }

  std::vector<double> raw_data(num_rows * num_cols);
  const std::array<hsize_t, 2> memspace_size{{num_rows, num_cols}};
  const hid_t memspace_id =
      H5Screate_simple(2, memspace_size.data(), memspace_size.data());
  CHECK_H5(memspace_id, "Failed to create memory space");
  CHECK_H5(H5Dread(dataset_id_, h5_type<double>(), memspace_id, dataspace_id,
                   h5::h5p_default(), raw_data.data()),
           "Failed to read data subset");

  CHECK_H5(H5Sclose(memspace_id), "Failed to close memory space");
  CHECK_H5(H5Sclose(dataspace_id), "Failed to close dataspace");
  return vector_to_matrix(raw_data,
                          std::array<hsize_t, 2>{{num_rows, num_cols}});
}
}  // namespace h5
