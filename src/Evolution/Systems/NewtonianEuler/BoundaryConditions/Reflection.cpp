// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/Reflection.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::BoundaryConditions {
template <size_t Dim>
Reflection<Dim>::Reflection(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
Reflection<Dim>::get_clone() const {
  return std::make_unique<Reflection>(*this);
}

template <size_t Dim>
void Reflection<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
}

template <size_t Dim>
std::optional<std::string> Reflection<Dim>::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        momentum_density,
    const gsl::not_null<Scalar<DataVector>*> energy_density,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_mass_density,
    const gsl::not_null<tnsr::IJ<DataVector, Dim, Frame::Inertial>*>
        flux_momentum_density,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        flux_energy_density,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,

    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, Dim, Frame::Inertial>&
        outward_directed_normal_covector,

    const Scalar<DataVector>& interior_mass_density,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_velocity,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_pressure) const {
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>> buffer{
      get(interior_mass_density).size()};
  auto& normal_dot_velocity = get<::Tags::TempScalar<0>>(buffer);
  dot_product(make_not_null(&normal_dot_velocity),
              outward_directed_normal_covector, interior_velocity);
  for (size_t i = 0; i < Dim; i++) {
    (*velocity).get(i) =
        interior_velocity.get(i) - 2.0 * get(normal_dot_velocity) *
                                       outward_directed_normal_covector.get(i);
  }
  if (face_mesh_velocity.has_value()) {
    auto& normal_dot_mesh_velocity = get<::Tags::TempScalar<1>>(buffer);
    dot_product(make_not_null(&normal_dot_mesh_velocity),
                outward_directed_normal_covector, face_mesh_velocity.value());
    for (size_t i = 0; i < Dim; i++) {
      (*velocity).get(i) += 2.0 * get(normal_dot_mesh_velocity) *
                            outward_directed_normal_covector.get(i);
    }
  }

  *specific_internal_energy = interior_specific_internal_energy;

  ConservativeFromPrimitive<Dim>::apply(
      mass_density, momentum_density, energy_density, interior_mass_density,
      *velocity, interior_specific_internal_energy);
  ComputeFluxes<Dim>::apply(flux_mass_density, flux_momentum_density,
                            flux_energy_density, *momentum_density,
                            *energy_density, *velocity, interior_pressure);

  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID Reflection<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) template class Reflection<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#undef DIM
}  // namespace NewtonianEuler::BoundaryConditions
