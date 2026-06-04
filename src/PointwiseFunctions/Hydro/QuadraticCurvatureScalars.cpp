// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/QuadraticCurvatureScalars.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/GeneralRelativity/QuadraticCurvatureScalars.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

template <typename DataType>
void kretschmann_scalar(const gsl::not_null<Scalar<DataType>*> result,
                        const Scalar<DataType>& weyl_electric_scalar,
                        const Scalar<DataType>& weyl_magnetic_scalar,
                        const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
                        const tnsr::aa<DataType, 3>& ricci_tensor,
                        const Scalar<DataType>& ricci_scalar) {
  Scalar<DataType> kretschmann_scalar_in_vacuum{
      get_size(get<0, 0>(inverse_spacetime_metric))};
  gr::kretschmann_scalar_in_vacuum(make_not_null(&kretschmann_scalar_in_vacuum),
                                   weyl_electric_scalar, weyl_magnetic_scalar);

  tnsr::AA<DataType, 3> ricci_tensor_upper{
      get_size(get<0, 0>(inverse_spacetime_metric))};
  tenex::evaluate<ti::A, ti::B>(make_not_null(&ricci_tensor_upper),
                                ricci_tensor(ti::c, ti::d) *
                                    inverse_spacetime_metric(ti::A, ti::C) *
                                    inverse_spacetime_metric(ti::B, ti::D));

  tenex::evaluate<>(result, kretschmann_scalar_in_vacuum() +
                                2. * ricci_tensor(ti::a, ti::b) *
                                    ricci_tensor_upper(ti::A, ti::B) -
                                square(ricci_scalar()) / 3.);
}

template <typename DataType>
Scalar<DataType> kretschmann_scalar(
    const Scalar<DataType>& weyl_electric_scalar,
    const Scalar<DataType>& weyl_magnetic_scalar,
    const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
    const tnsr::aa<DataType, 3>& ricci_tensor,
    const Scalar<DataType>& ricci_scalar) {
  Scalar<DataType> result{get_size(get<0, 0>(inverse_spacetime_metric))};
  kretschmann_scalar(make_not_null(&result), weyl_electric_scalar,
                     weyl_magnetic_scalar, inverse_spacetime_metric,
                     ricci_tensor, ricci_scalar);
  return result;
}

}  // namespace hydro

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                    \
  template Scalar<DTYPE(data)> hydro::kretschmann_scalar(       \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,          \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar,          \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric, \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,             \
      const Scalar<DTYPE(data)>& ricci_scalar);                 \
  template void hydro::kretschmann_scalar(                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,         \
      const Scalar<DTYPE(data)>& weyl_electric_scalar,          \
      const Scalar<DTYPE(data)>& weyl_magnetic_scalar,          \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric, \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,             \
      const Scalar<DTYPE(data)>& ricci_scalar);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
