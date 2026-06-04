// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/WeylElectric.hpp"

#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {
template <typename DataType>
tnsr::ii<DataType, 3> weyl_electric(
    const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
    const tnsr::AA<DataType, 3>& stress_energy,
    const tnsr::aa<DataType, 3>& ricci_tensor,
    const Scalar<DataType>& ricci_scalar,
    const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
    const tnsr::aa<DataType, 3>& induced_spatial_metric) {
  tnsr::ii<DataType, 3> result{get_size(get<0, 0>(stress_energy))};
  weyl_electric(make_not_null(&result), vacuum_weyl_electric, stress_energy,
                ricci_tensor, ricci_scalar, inverse_spacetime_metric,
                induced_spatial_metric);
  return result;
}

template <typename DataType>
void weyl_electric(const gsl::not_null<tnsr::ii<DataType, 3>*> weyl_electric,
                   const tnsr::ii<DataType, 3>& vacuum_weyl_electric,
                   const tnsr::AA<DataType, 3>& stress_energy,
                   const tnsr::aa<DataType, 3>& ricci_tensor,
                   const Scalar<DataType>& ricci_scalar,
                   const tnsr::AA<DataType, 3>& inverse_spacetime_metric,
                   const tnsr::aa<DataType, 3>& induced_spatial_metric) {
  set_number_of_grid_points(weyl_electric, stress_energy);

  // gamma_aB(a, B) = γ_a^B = g^{BC} γ_{aC}
  auto gamma_aB =
      make_with_value<tnsr::aB<DataType, 3>>(get<0, 0>(stress_energy), 0.0);
  ::tenex::evaluate<ti::a, ti::B>(make_not_null(&gamma_aB),
                                  inverse_spacetime_metric(ti::B, ti::C) *
                                      induced_spatial_metric(ti::a, ti::c));

  // gamma_trace = γ^{ab} R_{ab} = g^{AB} γ_B^C R_{AC}
  auto gamma_trace =
      make_with_value<Scalar<DataType>>(get<0, 0>(stress_energy), 0.0);
  tenex::evaluate<>(make_not_null(&gamma_trace),
                    inverse_spacetime_metric(ti::A, ti::B) *
                        gamma_aB(ti::b, ti::C) * ricci_tensor(ti::a, ti::c));

  // Assemble directly into the spatial output components, avoiding the wasted
  // time-component evaluations of computing the full 4D spacetime tensor.
  // E_{ij} = E_{ij}^{vac}
  //        - 1/2 (γ_{i+1}^a γ_{j+1}^b + γ_{(i+1)(j+1)} γ^{ab}) R_{ab}
  //        + 1/3 γ_{(i+1)(j+1)} R
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      weyl_electric->get(i, j) = vacuum_weyl_electric.get(i, j);
      for (size_t a = 0; a < 4; ++a) {
        for (size_t b = 0; b < 4; ++b) {
          weyl_electric->get(i, j) -= 0.5 * gamma_aB.get(i + 1, a) *
                                      gamma_aB.get(j + 1, b) *
                                      ricci_tensor.get(a, b);
        }
      }
      weyl_electric->get(i, j) +=
          induced_spatial_metric.get(i + 1, j + 1) *
          ((1.0 / 3.0) * get(ricci_scalar) - 0.5 * get(gamma_trace));
    }
  }
}
}  // namespace hydro

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template tnsr::ii<DTYPE(data), 3> hydro::weyl_electric(           \
      const tnsr::ii<DTYPE(data), 3>& vacuum_weyl_electric,         \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,                \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,                 \
      const Scalar<DTYPE(data)>& ricci_scalar,                      \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric,     \
      const tnsr::aa<DTYPE(data), 3>& induced_spatial_metric);      \
  template void hydro::weyl_electric(                               \
      const gsl::not_null<tnsr::ii<DTYPE(data), 3>*> weyl_electric, \
      const tnsr::ii<DTYPE(data), 3>& vacuum_weyl_electric,         \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,                \
      const tnsr::aa<DTYPE(data), 3>& ricci_tensor,                 \
      const Scalar<DTYPE(data)>& ricci_scalar,                      \
      const tnsr::AA<DTYPE(data), 3>& inverse_spacetime_metric,     \
      const tnsr::aa<DTYPE(data), 3>& induced_spatial_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
