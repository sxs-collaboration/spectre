// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes the second time derivative of the spacetime metric from the
 * generalized harmonic variables, lapse, shift, and the spacetime unit normal
 * 1-form.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -n^c \partial_c g_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i g_{ab} \f$.
 *
 * Using eq.(35) of \cite Lindblom2005qh (with \f$\gamma_1 = -1\f$) the first
 * time derivative of the spacetime metric may be expressed in terms of the
 * above variables:
 *
 * \f[
 * \partial_0 g_{ab} = - \alpha \Pi_{ab} + \beta^k \Phi_{kab}
 * \f]
 *
 * As such, its second time derivative is simply the following:
 *
 * \f[
 * \partial^2_0 g_{ab}
 *   = - (\partial_0 \alpha) \Pi_{ab} - \alpha \partial_0 \Pi_{ab}
 *     + (\partial_0 \beta^k) \Phi_{kab} + \beta^k \partial_0 \Phi_{kab}
 * \f]
 *
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void second_time_deriv_of_spacetime_metric(
    gsl::not_null<tnsr::aa<DataType, SpatialDim, Frame>*> d2t2_spacetime_metric,
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& dt_phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::aa<DataType, SpatialDim, Frame>& dt_pi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::aa<DataType, SpatialDim, Frame> second_time_deriv_of_spacetime_metric(
    const Scalar<DataType>& lapse, const Scalar<DataType>& dt_lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::I<DataType, SpatialDim, Frame>& dt_shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& dt_phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::aa<DataType, SpatialDim, Frame>& dt_pi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get second time derivative of the spacetime metric
 *        from generalized harmonic and geometric variables
 *
 * \details See `second_time_deriv_of_spacetime_metric()`. Can be retrieved
 * using `gr::Tags::SpacetimeMetric` wrapped in `Tags::dt<Tags::dt>>`.
 */
template <size_t SpatialDim, typename Frame>
struct SecondTimeDerivOfSpacetimeMetricCompute
    : ::Tags::dt<
          ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, SpatialDim, Frame>>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 ::Tags::dt<gr::Tags::Lapse<DataVector>>,
                 gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                 ::Tags::dt<gr::Tags::Shift<DataVector, SpatialDim, Frame>>,
                 Phi<DataVector, SpatialDim, Frame>,
                 ::Tags::dt<Phi<DataVector, SpatialDim, Frame>>,
                 Pi<DataVector, SpatialDim, Frame>,
                 ::Tags::dt<Pi<DataVector, SpatialDim, Frame>>>;

  using base = ::Tags::dt<
      ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, SpatialDim, Frame>>>;
  using return_type = typename base::type;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::aa<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const Scalar<DataVector>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&)>(
      &second_time_deriv_of_spacetime_metric<DataVector, SpatialDim, Frame>);
};
}  // namespace Tags
}  // namespace gh
