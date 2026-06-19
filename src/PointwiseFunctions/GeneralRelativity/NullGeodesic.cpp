// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/NullGeodesic.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType, size_t Dim, typename Frame>
void photon_geodesic_equation_with_constraint(
    const gsl::not_null<tnsr::I<DataType, Dim, Frame>*> dt_x,
    const gsl::not_null<tnsr::i<DataType, Dim, Frame>*> dt_pi,
    const gsl::not_null<Scalar<DataType>*> current_p0,
    const gsl::not_null<Scalar<DataType>*> current_dt_lnp0,
    const tnsr::I<DataType, Dim, Frame>& /*x*/,
    const tnsr::i<DataType, Dim, Frame>& pi,
    const Scalar<DataType>& lapse,
    const tnsr::i<DataType, Dim, Frame>& deriv_lapse,
    const tnsr::I<DataType, Dim, Frame>& shift,
    const tnsr::iJ<DataType, Dim, Frame>& deriv_shift,
    const tnsr::II<DataType, Dim, Frame>& inv_spatial_metric,
    const tnsr::ijj<DataType, Dim, Frame>& deriv_spatial_metric,
    [[maybe_unused]] const tnsr::ii<DataType, Dim, Frame>& extrinsic_curvature
    ) {

  // -------------------------------------------------------------------------
  // 1. Compute the contravariant spatial momentum pi^i = \gamma^{ij} pi_j
  // -------------------------------------------------------------------------
  // Use make_with_value
  // to allocate independent memory to avoid polluting dt_x with direct writes
  auto upper_pi = make_with_value<tnsr::I<DataType, Dim, Frame>>(lapse, 0.0);
  raise_or_lower_index(make_not_null(&upper_pi), pi, inv_spatial_metric);

  // -------------------------------------------------------------------------
  // 2. algebraic constraint: Compute the contravariant time component
  // p^0 satisfying the speed of light condition
  // Formula: p^0 = \sqrt{\gamma^{ij} pi_i pi_j} / \alpha
  // -------------------------------------------------------------------------
  auto pi_squared = make_with_value<Scalar<DataType>>(lapse, 0.0);
  tenex::evaluate(make_not_null(&pi_squared), upper_pi(ti::I) * pi(ti::i));
  tenex::evaluate(current_p0, sqrt(pi_squared()) / lapse());

  const auto& p0 = *current_p0;

  // -------------------------------------------------------------------------
  // 3. First-order rate of change of energy
  // -------------------------------------------------------------------------
  // Because we directly evolve the complete p_i,
  // the energy E = \alpha p^0 - \beta^i p_i is automatically conserved
  *current_dt_lnp0 = make_with_value<Scalar<DataType>>(lapse, 0.0);

  // -------------------------------------------------------------------------
  // 4. Compute the evolution of the photon position
  // (with respect to coordinate time t)
  // Formula: dx^i / dt = p^i / p^0 - \beta^i
  // -------------------------------------------------------------------------
  // Note: The template parameter <ti::I> must be explicitly specified
  // for the Left-Hand Side (LHS)
  tenex::evaluate<ti::I>(dt_x, upper_pi(ti::I) / p0() - shift(ti::I));

  // -------------------------------------------------------------------------
  // 5. Compute the evolution of the photon's covariant momentum
  // (with respect to coordinate time t)
  // Formula: dp_i / dt = - \alpha (\partial_i \alpha) p^0
  //                      + p_k \partial_i \beta^k
  //                      + \frac{1}{2 p^0} p^m p^n \partial_i \gamma_{mn}
  // -------------------------------------------------------------------------
  // Note: The template parameter <ti::i> must be explicitly specified
  // for the Left-Hand Side (LHS)
  tenex::evaluate<ti::i>(
      dt_pi,
      - lapse() * deriv_lapse(ti::i) * p0()
      + pi(ti::k) * deriv_shift(ti::i, ti::K)
      + (0.5 / p0()) * upper_pi(ti::M) * upper_pi(ti::N)
      * deriv_spatial_metric(ti::i, ti::m, ti::n));
}

// Explicit Instantiation
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data)   BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                  \
  template void photon_geodesic_equation_with_constraint(                     \
      gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*> dt_x,      \
      gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*> dt_pi,     \
      gsl::not_null<Scalar<DTYPE(data)>*> current_p0,                         \
      gsl::not_null<Scalar<DTYPE(data)>*> current_dt_lnp0,                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& x,                  \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& pi,                 \
      const Scalar<DTYPE(data)>& lapse,                                       \
      const tnsr::i<DTYPE(data), DIM(data), FRAME(data)>& deriv_lapse,        \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,              \
      const tnsr::iJ<DTYPE(data), DIM(data), FRAME(data)>& deriv_shift,       \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>& inv_spatial_metric,\
      const tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>&                   \
          deriv_spatial_metric,                                               \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>&                    \
          extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double), (3),(Frame::Inertial))

#undef DTYPE
#undef DIM
#undef FRAME
#undef INSTANTIATE

}  // namespace gr
