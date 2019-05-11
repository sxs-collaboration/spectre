// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

/// \cond
class Matrix;
template <size_t>
class Mesh;
/// \endcond

namespace Spectral {
/*!
 * \ingroup SpectralGroup
 * \brief Matrices for filtering spectral coefficients.
 */
namespace filtering {
/*!
 * \brief Returns a `Matrix` by which to multiply the nodal coefficients to
 * apply a stable exponential filter.
 *
 * The exponential filter rescales the modal coefficients according to:
 *
 * \f{align*}{
 *  c_i\to c_i \exp\left[-\alpha \left(\frac{i}{N}\right)^{2m}\right]
 * \f}
 *
 * where \f$c_i\f$ are the zero-indexed modal coefficients, \f$N\f$ is the basis
 * order, \f$\alpha\f$ determines how much the coefficients are rescaled, and
 * \f$m\f$ determines how aggressive/broad the filter is (lower values means
 * filtering more coefficients). Setting \f$\alpha=36\f$ results in setting the
 * highest coefficient to machine precision, effectively zeroing it out.
 *
 * \note The filter matrix is not cached by the function because it depends on a
 * double, an integer, and the mesh, which could make caching very memory
 * intensive. The caller of this function is responsible for determining whether
 * or not the matrix should be cached.
 */
Matrix exponential_filter(const Mesh<1>& mesh, double alpha,
                          unsigned half_power) noexcept;
}  // namespace filtering
}  // namespace Spectral
