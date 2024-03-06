// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/BoundaryConditions/DirichletAnalytic.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>

#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/NewtonianEuler/AllSolutions.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace NewtonianEuler::BoundaryConditions {
template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(const DirichletAnalytic<Dim>& rhs)
    : BoundaryCondition<Dim>{dynamic_cast<const BoundaryCondition<Dim>&>(rhs)},
      analytic_prescription_(rhs.analytic_prescription_->get_clone()) {}

template <size_t Dim>
DirichletAnalytic<Dim>& DirichletAnalytic<Dim>::operator=(
    const DirichletAnalytic<Dim>& rhs) {
  if (&rhs == this) {
    return *this;
  }
  analytic_prescription_ = rhs.analytic_prescription_->get_clone();
  return *this;
}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription)
    : analytic_prescription_(std::move(analytic_prescription)) {}

template <size_t Dim>
DirichletAnalytic<Dim>::DirichletAnalytic(CkMigrateMessage* const msg)
    : BoundaryCondition<Dim>(msg) {}

template <size_t Dim>
std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalytic<Dim>::get_clone() const {
  return std::make_unique<DirichletAnalytic>(*this);
}

template <size_t Dim>
void DirichletAnalytic<Dim>::pup(PUP::er& p) {
  BoundaryCondition<Dim>::pup(p);
  p | analytic_prescription_;
}

template <size_t Dim>
std::optional<std::string> DirichletAnalytic<Dim>::dg_ghost(
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

    const std::optional<
        tnsr::I<DataVector, Dim, Frame::Inertial>>& /*face_mesh_velocity*/,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& /*normal_covector*/,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const double time) const {
  auto boundary_values = call_with_dynamic_type<
      tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataVector>,
                          hydro::Tags::SpatialVelocity<DataVector, Dim>,
                          hydro::Tags::Pressure<DataVector>,
                          hydro::Tags::SpecificInternalEnergy<DataVector>>,
      NewtonianEuler::InitialData::initial_data_list<Dim>>(
      analytic_prescription_.get(),
      [&coords, &time](const auto* const initial_data) {
        if constexpr (is_analytic_solution_v<
                          std::decay_t<decltype(*initial_data)>>) {
          return initial_data->variables(
              coords, time,
              tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, Dim>,
                         hydro::Tags::Pressure<DataVector>,
                         hydro::Tags::SpecificInternalEnergy<DataVector>>{});

        } else {
          (void)time;
          return initial_data->variables(
              coords,
              tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                         hydro::Tags::SpatialVelocity<DataVector, Dim>,
                         hydro::Tags::Pressure<DataVector>,
                         hydro::Tags::SpecificInternalEnergy<DataVector>>{});
        }
      });

  *mass_density =
      get<hydro::Tags::RestMassDensity<DataVector>>(boundary_values);
  *velocity =
      get<hydro::Tags::SpatialVelocity<DataVector, Dim>>(boundary_values);
  *specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(boundary_values);

  ConservativeFromPrimitive<Dim>::apply(mass_density, momentum_density,
                                        energy_density, *mass_density,
                                        *velocity, *specific_internal_energy);
  ComputeFluxes<Dim>::apply(
      flux_mass_density, flux_momentum_density, flux_energy_density,
      *momentum_density, *energy_density, *velocity,
      get<hydro::Tags::Pressure<DataVector>>(boundary_values));

  return {};
}

template <size_t Dim>
// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalytic<Dim>::my_PUP_ID = 0;

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data) template class DirichletAnalytic<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace NewtonianEuler::BoundaryConditions
