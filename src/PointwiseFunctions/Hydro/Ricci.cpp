// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Hydro/Ricci.hpp"

#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace hydro {
template <typename DataType>
void ricci_in_gr(const gsl::not_null<tnsr::aa<DataType, 3>*> result,
                 const tnsr::AA<DataType, 3>& stress_energy,
                 const tnsr::aa<DataType, 3>& spacetime_metric) {
  set_number_of_grid_points(result, spacetime_metric);
  const Scalar<DataType> trace_stress_energy =
      trace(stress_energy, spacetime_metric);
  ::tenex::evaluate<ti::a, ti::b>(
      result,
      8. * M_PI *
          (stress_energy(ti::C, ti::D) * spacetime_metric(ti::a, ti::c) *
               spacetime_metric(ti::b, ti::d) -
           0.5 * trace_stress_energy() * spacetime_metric(ti::a, ti::b)));
}

template <typename DataType>
tnsr::aa<DataType, 3> ricci_in_gr(
    const tnsr::AA<DataType, 3>& stress_energy,
    const tnsr::aa<DataType, 3>& spacetime_metric) {
  tnsr::aa<DataType, 3> result{get_size(get<0, 0>(spacetime_metric))};
  ricci_in_gr(make_not_null(&result), stress_energy, spacetime_metric);
  return result;
}
}  // namespace hydro

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                 \
  template void hydro::ricci_in_gr(                          \
      const gsl::not_null<tnsr::aa<DTYPE(data), 3>*> result, \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,         \
      const tnsr::aa<DTYPE(data), 3>& spacetime_metric);     \
  template tnsr::aa<DTYPE(data), 3> hydro::ricci_in_gr(      \
      const tnsr::AA<DTYPE(data), 3>& stress_energy,         \
      const tnsr::aa<DTYPE(data), 3>& spacetime_metric);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector))

#undef DTYPE
#undef INSTANTIATE
