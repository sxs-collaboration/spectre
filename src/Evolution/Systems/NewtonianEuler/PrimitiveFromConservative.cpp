// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {

template <size_t Dim>
void PrimitiveFromConservative<Dim>::apply(
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density) noexcept {
  const DataVector one_over_mass_density = 1.0 / get(mass_density);

  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * one_over_mass_density;
  }

  get(*specific_internal_energy) = one_over_mass_density * get(energy_density) -
                                   0.5 * get(dot_product(*velocity, *velocity));
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template struct NewtonianEuler::PrimitiveFromConservative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
