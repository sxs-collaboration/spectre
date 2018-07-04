// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Matrix.

#pragma once

#include "DataStructures/DenseMatrix.hpp"  // IWYU pragma: export

/*!
 * \ingroup DataStructuresGroup
 * \brief A dynamically sized matrix of `double`s with column-major storage.
 *
 * \note This is a thin wrapper around `blaze::DynamicMatrix`. Please refer to
 * [Blaze documentation](https://bitbucket.org/blaze-lib/blaze/wiki/Matrices)
 * for information on how to use it.
 */
class Matrix : public DenseMatrix<double, blaze::columnMajor> {
 public:
  // Inherit constructors
  using DenseMatrix<double, blaze::columnMajor>::DenseMatrix;
};
