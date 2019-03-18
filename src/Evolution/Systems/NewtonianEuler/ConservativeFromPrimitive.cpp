// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"

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
void ConservativeFromPrimitive<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,
    const Scalar<DataVector>& mass_density,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& specific_internal_energy) noexcept {
  get(*mass_density_cons) = get(mass_density);

  for (size_t i = 0; i < Dim; ++i) {
    momentum_density->get(i) = get(mass_density) * velocity.get(i);
  }

  get(*energy_density) =
      get(mass_density) * (0.5 * get(dot_product(velocity, velocity)) +
                           get(specific_internal_energy));
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) \
  template struct NewtonianEuler::ConservativeFromPrimitive<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond
