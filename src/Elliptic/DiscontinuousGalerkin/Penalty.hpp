// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"

namespace elliptic::dg {

/*!
 * \brief The penalty factor in internal penalty fluxes
 *
 * The penalty factor is computed as
 *
 * \f{equation}
 * \sigma = C \frac{N_\text{points}^2}{h}
 * \f}
 *
 * where \f$N_\text{points} = 1 + N_p\f$ is the number of points (or one plus
 * the polynomial degree) and \f$h\f$ is a measure of the element size. Both
 * quantities are taken perpendicular to the face of the DG element that the
 * penalty is being computed on. \f$C\f$ is the "penalty parameter". For details
 * see section 7.2 in \cite HesthavenWarburton.
 */
DataVector penalty(const DataVector& element_size, size_t num_points,
                   double penalty_parameter) noexcept;

}  // namespace elliptic::dg
