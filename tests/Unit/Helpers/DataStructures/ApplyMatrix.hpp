// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
class DataVector;
class Matrix;
/// \endcond

/// Multiply a matrix and a vector
DataVector apply_matrix(const Matrix& m, const DataVector& v);
