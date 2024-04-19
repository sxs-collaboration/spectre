
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

template <size_t Dim>
tnsr::A<double, Dim> dt_self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi, const tnsr::a<double, Dim>& dt_d_psi,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const tnsr::AA<double, Dim>& dt_inverse_metric) {
  return tenex::evaluate<ti::A>(
      particle_charge / particle_mass *
      ((dt_inverse_metric(ti::A, ti::B) +
        four_velocity(ti::A) * dt_four_velocity(ti::B) +
        dt_four_velocity(ti::A) * four_velocity(ti::B)) *
           d_psi(ti::b) +
       (inverse_metric(ti::A, ti::B) +
        four_velocity(ti::A) * four_velocity(ti::B)) *
           dt_d_psi(ti::b)));
}

template <size_t Dim>
tnsr::A<double, Dim> dt2_self_force_per_mass(
    const tnsr::a<double, Dim>& d_psi, const tnsr::a<double, Dim>& dt_d_psi,
    const tnsr::a<double, Dim>& dt2_d_psi,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity,
    const tnsr::A<double, Dim>& dt2_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, Dim>& inverse_metric,
    const tnsr::AA<double, Dim>& dt_inverse_metric,
    const tnsr::AA<double, Dim>& dt2_inverse_metric) {
  return tenex::evaluate<ti::A>(
      particle_charge / particle_mass *
      ((dt2_inverse_metric(ti::A, ti::B) +
        dt2_four_velocity(ti::A) * four_velocity(ti::B) +
        dt2_four_velocity(ti::B) * four_velocity(ti::A) +
        2. * dt_four_velocity(ti::A) * dt_four_velocity(ti::B)) *
           d_psi(ti::b) +
       2. * dt_d_psi(ti::b) *
           (dt_inverse_metric(ti::A, ti::B) +
            dt_four_velocity(ti::A) * four_velocity(ti::B) +
            dt_four_velocity(ti::B) * four_velocity(ti::A)) +
       dt2_d_psi(ti::b) * (inverse_metric(ti::A, ti::B) +
                           four_velocity(ti::A) * four_velocity(ti::B))));
}

template <size_t Dim>
tnsr::A<double, Dim> Du_self_force_per_mass(
    const tnsr::A<double, Dim>& self_force,
    const tnsr::A<double, Dim>& dt_self_force,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::Abb<double, Dim>& christoffel) {
  return tenex::evaluate<ti::A>(dt_self_force(ti::A) * get<0>(four_velocity) +
                                christoffel(ti::A, ti::b, ti::c) *
                                    four_velocity(ti::B) * self_force(ti::C));
}

template <size_t Dim>
tnsr::A<double, Dim> dt_Du_self_force_per_mass(
    const tnsr::A<double, Dim>& self_force,
    const tnsr::A<double, Dim>& dt_self_force,
    const tnsr::A<double, Dim>& dt2_self_force,
    const tnsr::A<double, Dim>& four_velocity,
    const tnsr::A<double, Dim>& dt_four_velocity,
    const tnsr::Abb<double, Dim>& christoffel,
    const tnsr::Abb<double, Dim>& dt_christoffel) {
  return tenex::evaluate<ti::A>(
      dt2_self_force(ti::A) * get<0>(four_velocity) +
      dt_self_force(ti::A) * get<0>(dt_four_velocity) +
      dt_christoffel(ti::A, ti::b, ti::c) * four_velocity(ti::B) *
          self_force(ti::C) +
      christoffel(ti::A, ti::b, ti::c) * dt_four_velocity(ti::B) *
          self_force(ti::C) +
      christoffel(ti::A, ti::b, ti::c) * four_velocity(ti::B) *
          dt_self_force(ti::C));
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

template tnsr::A<double, 3> dt_self_force_per_mass(
    const tnsr::a<double, 3>& d_psi, const tnsr::a<double, 3>& dt_d_psi,
    const tnsr::A<double, 3>& four_velocity,
    const tnsr::A<double, 3>& dt_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, 3>& inverse_metric,
    const tnsr::AA<double, 3>& dt_inverse_metric);

template tnsr::A<double, 3> dt2_self_force_per_mass(
    const tnsr::a<double, 3>& d_psi, const tnsr::a<double, 3>& dt_d_psi,
    const tnsr::a<double, 3>& dt2_d_psi,
    const tnsr::A<double, 3>& four_velocity,
    const tnsr::A<double, 3>& dt_four_velocity,
    const tnsr::A<double, 3>& dt2_four_velocity, double particle_charge,
    double particle_mass, const tnsr::AA<double, 3>& inverse_metric,
    const tnsr::AA<double, 3>& dt_inverse_metric,
    const tnsr::AA<double, 3>& dt2_inverse_metric);

template tnsr::A<double, 3> Du_self_force_per_mass(
    const tnsr::A<double, 3>& self_force,
    const tnsr::A<double, 3>& dt_self_force,
    const tnsr::A<double, 3>& four_velocity,
    const tnsr::Abb<double, 3>& christoffel);

template tnsr::A<double, 3> dt_Du_self_force_per_mass(
    const tnsr::A<double, 3>& self_force,
    const tnsr::A<double, 3>& dt_self_force,
    const tnsr::A<double, 3>& dt2_self_force,
    const tnsr::A<double, 3>& four_velocity,
    const tnsr::A<double, 3>& dt_four_velocity,
    const tnsr::Abb<double, 3>& christoffel,
    const tnsr::Abb<double, 3>& dt_christoffel);
}  // namespace CurvedScalarWave::Worldtube
