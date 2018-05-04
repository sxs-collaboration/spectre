// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Matrix.

#pragma once

#include <blaze/math/DynamicMatrix.h>

#include "Utilities/Blaze.hpp"

namespace PUP {
class er;
}  // namespace PUP

// We are choosing row-major storage for the following reasons:
// - `Matrix` is nothrow move assignable and constructible.
// - It is the Blaze default.
// - The constructor takes arrays as rows x columns.
using Matrix = blaze::DynamicMatrix<double, blaze::rowMajor>;

/// Charm++ serialization
void operator|(PUP::er& p, Matrix& matrix);
