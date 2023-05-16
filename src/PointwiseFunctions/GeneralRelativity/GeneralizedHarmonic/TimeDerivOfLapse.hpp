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

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of lapse (\f$\alpha\f$) from the generalized
 *        harmonic variables, lapse, shift and the spacetime unit normal 1-form.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -n^c \partial_c g_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i g_{ab} \f$, and the operator
 * \f$D := \partial_0 - \beta^k \partial_k \f$. The time derivative of
 * \f$\alpha\f$ is then:
 *
 * \f{align*}
 *  \frac{1}{2} \alpha^2 n^a n^b \Pi_{ab}
 *       - \frac{1}{2} \alpha \beta^i n^a n^b \Phi_{iab}
 *  =& \frac{1}{2} \alpha^2 n^a n^b n^c \partial_c g_{ab}
 *       - \frac{1}{2} \alpha \beta^i (-(2/\alpha) \partial_i \alpha) \\
 *  =& \frac{1}{2} \alpha^2 [ \\
 *       &-(1/\alpha^3) D[\gamma_{jk} \beta^j \beta^k - \alpha^2] \\
 *       &- (\beta^j \beta^k / \alpha^3)D[\gamma_{jk}] \\
 *       &+ 2 (\beta^j / \alpha^3) D[\gamma_{jk} \beta^k] \\
 *       &+ (2 / \alpha^2)(\beta^i \partial_i \alpha)]] \\
 *  =& \frac{1}{2\alpha} [-D[\gamma_{jk}\beta^j\beta^k - \alpha^2]
 *       - \beta^j\beta^k D[\gamma_{jk}] + 2\alpha \beta^k\partial_k \alpha
 *       + 2\beta^j D[\gamma_{jk}\beta^k]] \\
 *  =& D[\alpha] + \beta^k\partial_k \alpha \\
 *  =& \partial_0 \alpha
 * \f}
 *
 * where the simplification done for \f$\partial_i \alpha\f$ is used to
 * substitute for the second term (\f$\frac{1}{2} \alpha \beta^i n^a n^b
 * \Phi_{iab}\f$).
 *
 * Thus,
 *
 * \f[
 *  \partial_0 \alpha =
 *      (\alpha/2)(\alpha n^a n^b \Pi_{ab} - \beta^i n^a n^b \Phi_{iab})
 * \f]
 *
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void time_deriv_of_lapse(
    gsl::not_null<Scalar<DataType>*> dt_lapse, const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);

template <typename DataType, size_t SpatialDim, typename Frame>
Scalar<DataType> time_deriv_of_lapse(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get time derivative of lapse (\f$\alpha\f$) from the
 *        generalized harmonic variables, lapse, shift and the spacetime unit
 *        normal 1-form.
 *
 * \details See `time_deriv_of_lapse()`. Can be retrieved using
 * `gr::Tags::Lapse` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivLapseCompute : ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                               db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                 gr::Tags::SpacetimeNormalVector<DataVector, SpatialDim, Frame>,
                 Phi<DataVector, SpatialDim, Frame>,
                 Pi<DataVector, SpatialDim, Frame>>;

  using return_type = Scalar<DataVector>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::A<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&)>(
      &time_deriv_of_lapse<DataVector, SpatialDim, Frame>);

  using base = ::Tags::dt<gr::Tags::Lapse<DataVector>>;
};
}  // namespace Tags
}  // namespace gh
