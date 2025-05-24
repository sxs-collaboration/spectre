// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeodesicEquation.hpp"

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType, size_t Dim, typename Frame>
void geodesic_equation(
    // Output time derivs
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> dt_x,
    const gsl::not_null<tnsr::i<DataType, Dim, Frame>*> dt_pi,
    const gsl::not_null<Scalar<DataType>*> dt_lnp0,
    // Current state
    const tnsr::I<DataType, Dim, Frame>& /*x*/,
    const tnsr::i<DataType, Dim, Frame>& pi, const Scalar<DataType>& /*lnp0*/,
    // Background spacetime
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::II<DataType, Dim, Frame>& inv_spatial_metric,
    const tnsr::iJJ<DataType, Dim, Frame>& deriv_inv_spatial_metric,
    const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature) {
  {
    // Scope where we reuse allocation of dt_x as upper_pi
    auto& upper_pi = *dt_x;
    raise_or_lower_index(make_not_null(&upper_pi), pi, inv_spatial_metric);
    tenex::evaluate(dt_lnp0, -deriv_lapse(ti::i) * upper_pi(ti::I) +
                                 lapse() * extrinsic_curvature(ti::i, ti::j) *
                                     upper_pi(ti::I) * upper_pi(ti::J));
  }
  // Complete computation of dt_x
  tenex::update<ti::I>(dt_x, lapse() * (*dt_x)(ti::I)-shift(ti::I));
  // Compute dt_pi
  tenex::evaluate<ti::i>(
      dt_pi, -deriv_lapse(ti::i) - (*dt_lnp0)() * pi(ti::i) +
                 deriv_shift(ti::i, ti::K) * pi(ti::k) -
                 0.5 * lapse() * deriv_inv_spatial_metric(ti::i, ti::J, ti::K) *
                     pi(ti::j) * pi(ti::k));
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                   \
  template void geodesic_equation(                                             \
      gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> dt_x,       \
      gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*> dt_pi,      \
      gsl::not_null<Scalar<DTYPE(data)>*> dt_lnp0,                             \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x,                   \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& pi,                  \
      const Scalar<DTYPE(data)>& lnp0, const Scalar<DTYPE(data)>& lapse,       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,               \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,        \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_spatial_metric, \
      const tnsr::iJJ<DTYPE(data), DIM(data), FRAME(data)>&                    \
          deriv_inv_spatial_metric,                                            \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                     \
          extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double), (3), (Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE

}  // namespace gr
