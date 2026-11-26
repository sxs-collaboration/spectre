// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr {

template <typename DataType, typename Frame>
void pontryagin_scalar(
    const gsl::not_null<Scalar<DataType>*> pontryagin_scalar,
    const tnsr::ii<DataType, 3, Frame>& weyl_electric,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric) {
  tenex::evaluate(pontryagin_scalar, 16.0 * weyl_electric(ti::i, ti::j) *
                                         inverse_spatial_metric(ti::J, ti::K) *
                                         weyl_magnetic(ti::k, ti::l) *
                                         inverse_spatial_metric(ti::L, ti::I));
}

template <typename DataType, typename Frame>
Scalar<DataType> pontryagin_scalar(
    const tnsr::ii<DataType, 3, Frame>& weyl_electric,
    const tnsr::ii<DataType, 3, Frame>& weyl_magnetic,
    const tnsr::II<DataType, 3, Frame>& inverse_spatial_metric) {
  Scalar<DataType> result{get_size(get<0, 0>(weyl_electric))};
  pontryagin_scalar(make_not_null(&result), weyl_electric, weyl_magnetic,
                    inverse_spatial_metric);
  return result;
}

template <typename DataType>
void kretschmann_scalar_in_vacuum(
    const gsl::not_null<Scalar<DataType>*> kretschmann_scalar,
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar) {
  // Note: Really this is just computing the parts of the Kretschmann scalar
  // that depend on the Weyl electric and magnetic scalars. When in vacuum, this
  // indeed is the pure vacuum contribution to the Kretschmann scalar. When not
  // in vacuum, the Weyl electric scalar technically includes some non-vacuum
  // contribution. However, in addition to this contribution, the Kretschmann
  // scalar has some separate, purely non-vacuum terms which are computed in the
  // Hydro namespace (see `hydro::kretschmann_scalar`).
  kretschmann_scalar->get() =
      8.0 * (weyl_electric_scalar.get() - weyl_magnetic_scalar.get());
}

template <typename DataType>
Scalar<DataType> kretschmann_scalar_in_vacuum(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar) {
  Scalar<DataType> kretschmann_scalar{get_size(get(weyl_electric_scalar))};
  kretschmann_scalar_in_vacuum(make_not_null(&kretschmann_scalar),
                               weyl_electric_scalar, weyl_magnetic_scalar);
  return kretschmann_scalar;
}

template <typename DataType>
void gauss_bonnet_scalar_in_vacuum(
    const gsl::not_null<Scalar<DataType>*> gb_scalar,
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar) {
  // The Gauss-Bonnet scalar is equivalent to the Kretschmann scalar in vacuum
  kretschmann_scalar_in_vacuum(gb_scalar, weyl_electric_scalar,
                               weyl_magnetic_scalar);
}

template <typename DataType>
Scalar<DataType> gauss_bonnet_scalar_in_vacuum(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar) {
  Scalar<DataType> gb_scalar{get_size(get(weyl_electric_scalar))};
  gauss_bonnet_scalar_in_vacuum(make_not_null(&gb_scalar), weyl_electric_scalar,
                                weyl_magnetic_scalar);
  return gb_scalar;
}

}  // namespace gr

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template Scalar<DTYPE(data)> gr::pontryagin_scalar(                       \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_electric,           \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_magnetic,           \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric); \
  template void gr::pontryagin_scalar(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> pontryagin_scalar,          \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_electric,           \
      const tnsr::ii<DTYPE(data), 3, FRAME(data)>& weyl_magnetic,           \
      const tnsr::II<DTYPE(data), 3, FRAME(data)>& inverse_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef FRAME
#undef DTYPE
#undef INSTANTIATE

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template Scalar<DTYPE(data)> gr::kretschmann_scalar_in_vacuum(    \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,              \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar);             \
  template void gr::kretschmann_scalar_in_vacuum(                   \
      const gsl::not_null<Scalar<DTYPE(data)>*> kretschmann_scalar, \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,              \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar);             \
  template Scalar<DTYPE(data)> gr::gauss_bonnet_scalar_in_vacuum(   \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,              \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar);             \
  template void gr::gauss_bonnet_scalar_in_vacuum(                  \
      const gsl::not_null<Scalar<DTYPE(data)>*> gb_scalar,          \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,              \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
