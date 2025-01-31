// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <complex>

/// \cond
class ComplexDataVector;
class DataVector;
class StepperErrorTolerances;
/// \endcond

/*!
 * \ingroup TimeGroup
 * \brief Calculate the pointwise worst error.
 *
 * \details For a `double`, or `std::complex<double>`, calculates
 *
 * \f{equation}
 *   \frac{e}{a + r \max(|v|, |v + e|)}
 * \f}
 *
 * where $v$ is \p values, $e$ is \p errors, and $a$ and $r$ are the
 * tolerances from \p tolerances.  For vector types, calculates the
 * largest error over all the points.
 *
 * Instantiated for `double`, `std::complex<double>`, `DataVector`,
 * and `ComplexDataVector`.
 */
template <typename T>
double largest_stepper_error(const T& values, const T& errors,
                             const StepperErrorTolerances& tolerances);
