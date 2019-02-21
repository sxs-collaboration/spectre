// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DenseMatrix.

#pragma once

#include <pup.h>  // IWYU pragma: keep

#include "Options/Options.hpp"
#include "Utilities/Blaze.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/system/Version.h>

/*!
 * \ingroup DataStructuresGroup
 * \brief A dynamically sized matrix of arbitrary type.
 *
 * \note This is a thin wrapper around `blaze::DynamicMatrix`. Please refer to
 * the [Blaze
 * documentation](https://bitbucket.org/blaze-lib/blaze/wiki/Matrices) for
 * information on how to use it.
 */
template <typename T, bool SO = blaze::defaultStorageOrder>
class DenseMatrix : public blaze::DynamicMatrix<T, SO> {
 public:
  // Inherit constructors
  using blaze::DynamicMatrix<T, SO>::DynamicMatrix;

  /// Charm++ serialization
  // clang-tidy: runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    auto rows = blaze::DynamicMatrix<T, SO>::rows();
    auto columns = blaze::DynamicMatrix<T, SO>::columns();
    p | rows;
    p | columns;
    if (p.isUnpacking()) {
      blaze::DynamicMatrix<T, SO>::resize(rows, columns);
    }
    if (cpp17::is_fundamental_v<T>) {
      PUParray(p, blaze::DynamicMatrix<T, SO>::data(), rows * columns);
    } else {
      for (size_t i = 0; i < rows * columns; i++) {
        p | blaze::DynamicMatrix<T, SO>::data()[i];
      }
    }
  }
};

template <typename T, bool SO>
struct create_from_yaml<DenseMatrix<T, SO>> {
  template <typename Metavariables>
  static DenseMatrix<T, SO> create(const Option& options) {
    const auto data = options.parse_as<std::vector<std::vector<T>>>();
    const size_t num_rows = data.size();
    size_t num_cols = 0;
    if (num_rows > 0) {
      num_cols = data[0].size();
    }
    DenseMatrix<T, SO> result(num_rows, num_cols);
    // We don't use an iterator over the matrix here to make the code
    // independent of the matrix storage order (see the [Blaze
    // documentation](https://bitbucket.org/
    // blaze-lib/blaze/wiki/Matrix%20Operations#!iterators)).
    for (size_t i = 0; i < num_rows; i++) {
      const auto& row = gsl::at(data, i);
      if (row.size() != num_cols) {
        PARSE_ERROR(options.context(),
                    "All matrix columns must have the same size.");
      }
      for (size_t j = 0; j < num_cols; j++) {
        result(i, j) = gsl::at(row, j);
      }
    }
    return result;
  }
};
