
// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

template <size_t Dim>
void self_force_acceleration(
    gsl::not_null<tnsr::I<double, Dim>*> self_force_acc,
    const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim>& psi_dipole,
    const tnsr::I<double, Dim>& particle_velocity, const double particle_charge,
    const double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const Scalar<double>& dilation_factor) {
  const double factor =
      particle_charge / particle_mass / square(get(dilation_factor));
  for (size_t i = 0; i < Dim; ++i) {
    self_force_acc->get(i) =
        (inverse_metric.get(i + 1, 0) -
         particle_velocity.get(i) * inverse_metric.get(0, 0)) *
        get(dt_psi_monopole) * factor;
    for (size_t j = 0; j < Dim; ++j) {
      self_force_acc->get(i) +=
          (inverse_metric.get(i + 1, j + 1) -
           particle_velocity.get(i) * inverse_metric.get(0, j + 1)) *
          psi_dipole.get(j) * factor;
    }
  }
}

template <size_t Dim>
tnsr::I<double, Dim> self_force_acceleration(
    const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim>& psi_dipole,
    const tnsr::I<double, Dim>& particle_velocity, const double particle_charge,
    const double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const Scalar<double>& dilation_factor) {
  tnsr::I<double, Dim> self_force_acc{};
  self_force_acceleration(make_not_null(&self_force_acc), dt_psi_monopole,
                          psi_dipole, particle_velocity, particle_charge,
                          particle_mass, inverse_metric, dilation_factor);
  return self_force_acc;
}

template <size_t Dim>
tnsr::A<double, Dim> self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi,
    const tnsr::A<double, Dim>& four_velocity, const double particle_charge,
    const double particle_mass, const tnsr::AA<double, Dim>& inverse_metric) {
  return tenex::evaluate<ti::B>(particle_charge / particle_mass * d_psi(ti::a) *
                                (inverse_metric(ti::A, ti::B) +
                                 four_velocity(ti::A) * four_velocity(ti::B)));
}

// Instantiations
template void self_force_acceleration(
    gsl::not_null<tnsr::I<double, 3>*> self_force_acc,
    const Scalar<double>& dt_psi_monopole, const tnsr::i<double, 3>& psi_dipole,
    const tnsr::I<double, 3>& particle_velocity, const double particle_charge,
    const double particle_mass, const tnsr::AA<double, 3>& inverse_metric,
    const Scalar<double>& dilation_factor);

template tnsr::I<double, 3> self_force_acceleration(
    const Scalar<double>& dt_psi_monopole, const tnsr::i<double, 3>& psi_dipole,
    const tnsr::I<double, 3>& particle_velocity, const double particle_charge,
    const double particle_mass, const tnsr::AA<double, 3>& inverse_metric,
    const Scalar<double>& dilation_factor);

template tnsr::A<double, 3> self_force_per_mass(
    const tnsr::a<double, 3>& d_psi, const tnsr::A<double, 3>& four_velocity,
    const double particle_charge, const double particle_mass,
    const tnsr::AA<double, 3>& inverse_metric);

}  // namespace CurvedScalarWave::Worldtube
