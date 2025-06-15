// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/ScalarWave/Potential.hpp"

#include "DataStructures/DataVector.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace ScalarWave {

template <size_t SpatialDim>
void potential(gsl::not_null<Scalar<DataVector>*> result,
               const Scalar<DataVector>& psi, const double& mass_squared) {
  get(*result) = 0.5 * mass_squared * square(get(psi));
}

template <size_t SpatialDim>
Scalar<DataVector> potential(const Scalar<DataVector>& psi,
                             const double& mass_squared) {
  Scalar<DataVector> result{get(psi).size()};
  potential<SpatialDim>(make_not_null(&result), psi, mass_squared);
  return result;
}

}  // namespace ScalarWave

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                         \
  template void ScalarWave::potential<DIM(data)>(                    \
      gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&, \
      const double&);                                                \
  template Scalar<DataVector> ScalarWave::potential<DIM(data)>(      \
      const Scalar<DataVector>&, const double&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
