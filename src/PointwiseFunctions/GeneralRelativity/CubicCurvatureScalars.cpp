// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/CubicCurvatureScalars.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType, size_t Dim, typename Frame>
void cubic_invariant_real(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) {
  const auto e_mixed = tenex::evaluate<ti::I, ti::j>(
      inverse_spatial_metric(ti::I, ti::K) * weyl_electric(ti::k, ti::j));
  const auto b_mixed = tenex::evaluate<ti::I, ti::j>(
      inverse_spatial_metric(ti::I, ti::K) * weyl_magnetic(ti::k, ti::j));

  // -1/6 tr(E^3) + 1/2 tr(E B^2)
  tenex::evaluate(
      result, -1.0 / 6.0 * e_mixed(ti::I, ti::j) * e_mixed(ti::J, ti::k) *
                      e_mixed(ti::K, ti::i) +
                  1.0 / 2.0 * e_mixed(ti::I, ti::j) * b_mixed(ti::J, ti::k) *
                      b_mixed(ti::K, ti::i));
}

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> cubic_invariant_real(
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) {
  Scalar<DataType> result{get<0, 0>(inverse_spatial_metric)};
  cubic_invariant_real(make_not_null(&result), weyl_electric, weyl_magnetic,
                       inverse_spatial_metric);
  return result;
}

template <typename DataType, size_t Dim, typename Frame>
void cubic_invariant_imag(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) {
  const auto e_mixed = tenex::evaluate<ti::I, ti::j>(
      inverse_spatial_metric(ti::I, ti::K) * weyl_electric(ti::k, ti::j));
  const auto b_mixed = tenex::evaluate<ti::I, ti::j>(
      inverse_spatial_metric(ti::I, ti::K) * weyl_magnetic(ti::k, ti::j));

  // 1/6 tr(B^3) - 1/2 tr(B E^2)
  tenex::evaluate(
      result, 1.0 / 6.0 * b_mixed(ti::I, ti::j) * b_mixed(ti::J, ti::k) *
                      b_mixed(ti::K, ti::i) -
                  1.0 / 2.0 * b_mixed(ti::I, ti::j) * e_mixed(ti::J, ti::k) *
                      e_mixed(ti::K, ti::i));
}

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> cubic_invariant_imag(
    const tnsr::ii<DataType, Dim, Frame>& weyl_electric,
    const tnsr::ii<DataType, Dim, Frame>& weyl_magnetic,
    const tnsr::II<DataType, Dim, Frame>& inverse_spatial_metric) {
  Scalar<DataType> result{get<0, 0>(inverse_spatial_metric)};
  cubic_invariant_imag(make_not_null(&result), weyl_electric, weyl_magnetic,
                       inverse_spatial_metric);
  return result;
}

}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                             \
  template void gr::cubic_invariant_real(                \
      const gsl::not_null<Scalar<DTYPE(data)>*>,         \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>&);     \
  template Scalar<DTYPE(data)> gr::cubic_invariant_real( \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>&);     \
  template void gr::cubic_invariant_imag(                \
      const gsl::not_null<Scalar<DTYPE(data)>*>,         \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>&);     \
  template Scalar<DTYPE(data)> gr::cubic_invariant_imag( \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>&,      \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
