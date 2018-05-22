// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class DenseMatrix.

#pragma once

#include <pup.h>  // IWYU pragma: keep

#include "Utilities/Blaze.hpp"
#include "Utilities/TypeTraits.hpp"

#include <blaze/math/DynamicMatrix.h>
#include <blaze/system/Version.h>

/*!
 * \ingroup DataStructuresGroup
 * \brief A dynamically sized matrix of arbitrary type.
 *
 * \note This is a thin wrapper around `blaze::DynamicMatrix`. Please refer to
 * [Blaze documentation](https://bitbucket.org/blaze-lib/blaze/wiki/Matrices)
 * for information on how to use it.
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
