// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace GeneralizedHarmonic {
// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of lapse (N) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -t^c \partial_c \psi_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i \psi_{ab} \f$, and the operator
 * \f$D := \partial_0 - N^k \partial_k \f$. The time derivative of N is then:
 * \f{align*}
 *  \frac{1}{2} N^2 t^a t^b \Pi_{ab} - \frac{1}{2} N N^i t^a t^b \Phi_{iab}
 *  =& \frac{1}{2} N^2 t^a t^b t^c \partial_c \psi_{ab}
 *       - \frac{1}{2} N N^i (-(2/N) \partial_i N) \\
 *  =& \frac{1}{2} N^2 [-(1/N^3) D[g_{jk} N^j N^k - N^2] \\
 *           &- (N^j N^k / N^3)D[g_{jk}] \\
 *           &+ 2 (N^j / N^3) D[g_{jk} N^k] + (2 / N^2)(N^i \partial_i N)] \\
 *  =& \frac{1}{2N} [-D[g_{jk}N^jN^k - N^2] - N^jN^k D[g_{jk}]
 *            + 2N N^k\partial_k N + 2N^j D[g_{jk}N^k]] \\
 *  =& D[N] + N^k\partial_k N \\
 *  =& \partial_0 N
 * \f}
 * where the simplification done for \f$\partial_i N\f$ is used to substitute
 * for the second term (\f$\frac{1}{2} N N^i t^a t^b \Phi_{iab}\f$).
 *
 * Thus,
 * \f[
 *  \partial_0 N = (N/2)(N t^a t^b \Pi_{ab} - N^i t^a t^b \Phi_{iab})
 * \f]
 */
template <size_t SpatialDim, typename Frame, typename DataType>
void time_deriv_of_lapse(
    gsl::not_null<Scalar<DataType>*> dt_lapse, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept;
// @}

namespace Tags {
/*!
 * \brief Compute item to get time derivative of lapse (N) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details See `time_deriv_of_lapse()`. Can be retrieved using
 * `gr::Tags::Lapse` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivLapseCompute : ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                               db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<SpatialDim, Frame, DataVector>,
                 gr::Tags::SpacetimeNormalVector<SpatialDim, Frame, DataVector>,
                 Phi<SpatialDim, Frame>, Pi<SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&) noexcept>(
      &time_deriv_of_lapse<SpatialDim, Frame>);

  using base = ::Tags::dt<gr::Tags::Lapse<DataVector>>;
};
}  // namespace Tags
}  // namespace GeneralizedHarmonic
