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
 *
 * \warning In Blaze 3.7, a combination of changes were made such that our use
 * of `DynamicMatrix` winds up with non-optional padding associated with the
 * length of vector registers on the target hardware, which often adds some
 * number of zeros to the fastest-varying dimension. To be compatible with
 * LAPACK calls in Blaze 3.7, it is necessary to use `blaze::columnMajor`
 * ordering (the current default storage order for this custom class) and to
 * provide `spacing()` to the LAPACK `LDA`, `LDB`, etc. parameters
 * whenever using the `data()` pointer from this class. This is a defect that is
 * expected to be solved by additional configuration options in future versions
 * of Blaze.
 */
template <typename T, bool SO = blaze::columnMajor>
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
    auto spacing = blaze::DynamicMatrix<T, SO>::spacing();
    if (cpp17::is_fundamental_v<T>) {
      PUParray(p, blaze::DynamicMatrix<T, SO>::data(), columns * spacing);
    } else {
      for (size_t i = 0; i < columns * spacing; i++) {
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
