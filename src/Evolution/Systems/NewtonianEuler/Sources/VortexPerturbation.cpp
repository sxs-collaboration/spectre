// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler::Sources {
template <>
void VortexPerturbation<2>::apply() const noexcept {}

template <>
void VortexPerturbation<3>::apply(
    const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const NewtonianEuler::Solutions::IsentropicVortex<3>& vortex,
    const tnsr::I<DataVector, 3>& x, const double time) const noexcept {
  const size_t number_of_grid_points = get<0>(x).size();
  const auto vortex_primitives = vortex.variables(
      x, time,
      tmpl::list<Tags::MassDensity<DataVector>, Tags::Velocity<DataVector, 3>,
                 Tags::SpecificInternalEnergy<DataVector>,
                 Tags::Pressure<DataVector>>{});
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempI<1, 3, Frame::Inertial>,
                 ::Tags::TempScalar<2>>>
      temp_buffer(number_of_grid_points);
  auto& vortex_mass_density_cons = get<::Tags::TempScalar<0>>(temp_buffer);
  auto& vortex_momentum_density =
      get<::Tags::TempI<1, 3, Frame::Inertial>>(temp_buffer);
  auto& vortex_energy_density = get<::Tags::TempScalar<2>>(temp_buffer);

  NewtonianEuler::ConservativeFromPrimitive<3>::apply(
      make_not_null(&vortex_mass_density_cons),
      make_not_null(&vortex_momentum_density),
      make_not_null(&vortex_energy_density),
      get<Tags::MassDensity<DataVector>>(vortex_primitives),
      get<Tags::Velocity<DataVector, 3>>(vortex_primitives),
      get<Tags::SpecificInternalEnergy<DataVector>>(vortex_primitives));

  // We save the precomputed value of dv_z/dz in source_mass_density_cons
  // in order to save an allocation
  get(*source_mass_density_cons) =
      vortex.perturbation_amplitude() *
      vortex.deriv_of_perturbation_profile(get<2>(x));

  for (size_t i = 0; i < 3; ++i) {
    source_momentum_density->get(i) =
        vortex_momentum_density.get(i) * get(*source_mass_density_cons);
  }
  source_momentum_density->get(2) *= 2.0;

  get(*source_energy_density) =
      (get(vortex_energy_density) +
       get(get<Tags::Pressure<DataVector>>(vortex_primitives)) +
       vortex_momentum_density.get(2) *
           get<2>(get<Tags::Velocity<DataVector, 3>>(
               vortex_primitives))) *
      get(*source_mass_density_cons);

  get(*source_mass_density_cons) *= get(vortex_mass_density_cons);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data) template struct VortexPerturbation<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace NewtonianEuler::Sources
/// \endcond
