// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/IterateAccelerationTerms.hpp"

#include <cstddef>
#include <optional>
#include <tuple>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/KerrSchildDerivatives.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/Strahlkorper/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
void IterateAccelerationTerms::apply(
    gsl::not_null<Scalar<DataVector>*> acceleration_terms,
    const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
    const tuples::TaggedTuple<
        gr::Tags::SpacetimeMetric<double, Dim>,
        gr::Tags::InverseSpacetimeMetric<double, Dim>,
        gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
        gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
        Tags::TimeDilationFactor>& background,
    const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
    const Scalar<double>& psi_monopole, const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole,
    const tnsr::i<double, Dim, Frame::Inertial>& dt_psi_dipole,
    const double charge, const std::optional<double> mass, const double time,
    const std::optional<double> turn_on_time,
    const std::optional<double> turn_on_interval, const size_t iteration) {
  // size of the data that is sent to the elements to compute the acceleration
  // terms of the puncture field. Consists of  the x, y and z components of the
  // acceleration, as well the t, x and y components of the scalar self force
  // and three different derivatives of it.
  const size_t data_size = 15;
  acceleration_terms->get() = DataVector(data_size, 0.);
  auto acc = geodesic_acc;
  double roll_on = 0.;
  if (time > turn_on_time.value()) {
    const auto& particle_position = pos_vel.at(0);
    const auto& particle_velocity = pos_vel.at(1);
    const auto& imetric =
        get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background);
    const auto& metric =
        get<gr::Tags::SpacetimeMetric<double, Dim>>(background);
    const auto& christoffel =
        get<gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>>(background);
    const auto& trace_christoffel =
        get<gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>>(
            background);
    const auto di_imetric =
        spatial_derivative_inverse_ks_metric(particle_position);
    const auto dij_imetric =
        second_spatial_derivative_inverse_ks_metric(particle_position);
    const auto di_trace_christoffel =
        spatial_derivative_ks_contracted_christoffel(particle_position);
    const auto di_metric = spatial_derivative_ks_metric(metric, di_imetric);
    const auto dij_metric = second_spatial_derivative_metric(
        metric, di_metric, di_imetric, dij_imetric);
    const auto di_christoffel = spatial_derivative_christoffel(
        di_metric, dij_metric, imetric, di_imetric);
    const auto& u0 = get(get<Tags::TimeDilationFactor>(background));

    // The evolution of the mass is a second order effect so we only include it
    // starting at the 2nd iteration.
    const double evolved_mass = iteration == 1
                                    ? mass.value()
                                    : mass.value() - charge * get(psi_monopole);

    const auto self_force_acc = self_force_acceleration(
        dt_psi_monopole, psi_dipole, particle_velocity, charge, evolved_mass,
        imetric, get<Tags::TimeDilationFactor>(background));

    const double t_minus_turn_on = time - turn_on_time.value();
    roll_on = turn_on_function(t_minus_turn_on, turn_on_interval.value());
    const double dt_roll_on =
        dt_turn_on_function(t_minus_turn_on, turn_on_interval.value());
    const double dt2_roll_on =
        dt2_turn_on_function(t_minus_turn_on, turn_on_interval.value());
    for (size_t i = 0; i < Dim; ++i) {
      acc.get(i) += roll_on * self_force_acc.get(i);
    }
    const tnsr::A<double, Dim> u{{u0, get<0>(particle_velocity) * u0,
                                  get<1>(particle_velocity) * u0,
                                  get<2>(particle_velocity) * u0}};
    const tnsr::a<double, Dim> d_psiR{{get(dt_psi_monopole), get<0>(psi_dipole),
                                       get<1>(psi_dipole), get<2>(psi_dipole)}};

    // Eq.(55) of Wittek:2024gxn
    double dt2_psiR = get<0>(trace_christoffel) * get<0>(d_psiR);
    for (size_t i = 0; i < Dim; ++i) {
      dt2_psiR += -2. * imetric.get(0, i + 1) * dt_psi_dipole.get(i) +
                  trace_christoffel.get(i + 1) * psi_dipole.get(i);
    }
    dt2_psiR /= get<0, 0>(imetric);

    tnsr::a<double, Dim> dt_d_psiR{{dt2_psiR, get<0>(dt_psi_dipole),
                                    get<1>(dt_psi_dipole),
                                    get<2>(dt_psi_dipole)}};
    for (size_t i = 0; i < Dim; ++i) {
      get<0>(dt_d_psiR) += particle_velocity.get(i) * dt_psi_dipole.get(i);
    }

    double dt_evolved_mass = 0.;
    double dt2_evolved_mass = 0.;
    double dt_mass_factor = 0.;
    double dt2_mass_factor = 0.;
    if (iteration > 1) {
      dt_evolved_mass = get(dt_psi_monopole);
      dt2_evolved_mass = dt2_psiR;
      for (size_t i = 0; i < Dim; ++i) {
        dt_evolved_mass += psi_dipole.get(i) * particle_velocity.get(i);
        dt2_evolved_mass +=
            2. * dt_psi_dipole.get(i) * particle_velocity.get(i) +
            psi_dipole.get(i) * acc.get(i);
      }
      dt_evolved_mass *= -charge;
      dt2_evolved_mass *= -charge;

      dt_mass_factor = -dt_evolved_mass / evolved_mass;
      dt2_mass_factor = (2. * dt_evolved_mass * dt_evolved_mass -
                         evolved_mass * dt2_evolved_mass) /
                        square(evolved_mass);
    }
    const auto dt_christoffel = tenex::evaluate<ti::A, ti::b, ti::c>(
        particle_velocity(ti::I) * di_christoffel(ti::i, ti::A, ti::b, ti::c));

    const auto dt_imetric = tenex::evaluate<ti::A, ti::B>(
        particle_velocity(ti::I) * di_imetric(ti::i, ti::A, ti::B));
    const auto dt2_imetric = tenex::evaluate<ti::A, ti::B>(
        particle_velocity(ti::I) * particle_velocity(ti::J) *
            dij_imetric(ti::i, ti::j, ti::A, ti::B) +
        acc(ti::I) * di_imetric(ti::i, ti::A, ti::B));
    // Eq.(51a) of Wittek:2024gxn
    const auto dt_u = tenex::evaluate<ti::A>(
        charge / evolved_mass / u0 * imetric(ti::A, ti::B) * d_psiR(ti::b) -
        christoffel(ti::A, ti::b, ti::c) * u(ti::B) * u(ti::C) / u0);
    // Eq.(51b) of Wittek:2024gxn
    const auto dt2_u = tenex::evaluate<ti::A>(
        charge / evolved_mass / u0 *
            (dt_imetric(ti::A, ti::B) * d_psiR(ti::b) +
             imetric(ti::A, ti::B) * dt_d_psiR(ti::b) +
             dt_mass_factor * imetric(ti::A, ti::B) * d_psiR(ti::b)) -
        (dt_christoffel(ti::A, ti::b, ti::c) * u(ti::B) * u(ti::C) +
         2. * christoffel(ti::A, ti::b, ti::c) * dt_u(ti::B) * u(ti::C) +
         get<0>(dt_u) * dt_u(ti::A)) /
            u0);
    const auto dt_u_rollon = tenex::evaluate<ti::A>(roll_on * dt_u(ti::A));
    const auto dt2_u_rollon = tenex::evaluate<ti::A>(dt_roll_on * dt_u(ti::A) +
                                                     roll_on * dt2_u(ti::A));

    // Eq.(56) of Wittek:2024gxn
    tnsr::i<double, Dim> d_dt2_psiR{0.};
    for (size_t i = 0; i < Dim; ++i) {
      d_dt2_psiR.get(i) +=
          -di_imetric.get(i, 0, 0) * dt2_psiR +
          di_trace_christoffel.get(i, 0) * get(dt_psi_monopole) +
          trace_christoffel.get(0) * dt_psi_dipole.get(i);
      for (size_t j = 0; j < Dim; ++j) {
        d_dt2_psiR.get(i) +=
            -2. * di_imetric.get(i, 0, j + 1) * dt_psi_dipole.get(j) +
            di_trace_christoffel.get(i, j + 1) * psi_dipole.get(j);
      }
      d_dt2_psiR.get(i) /= get<0, 0>(imetric);
    }

    // Eq.(57) of Wittek:2024gxn
    double dt3_psiR = trace_christoffel.get(0) * dt2_psiR;
    for (size_t i = 0; i < Dim; ++i) {
      dt3_psiR += -2. * imetric.get(0, i + 1) * d_dt2_psiR.get(i) +
                  trace_christoffel.get(i + 1) * dt_psi_dipole.get(i);
    }
    dt3_psiR /= get<0, 0>(imetric);
    tnsr::a<double, Dim> dt2_d_psiR;
    dt2_d_psiR.get(0) = dt3_psiR;
    for (size_t i = 0; i < Dim; ++i) {
      dt2_d_psiR.get(0) += 2. * d_dt2_psiR.get(i) * particle_velocity.get(i) +
                           dt_psi_dipole.get(i) * acc.get(i);
      dt2_d_psiR.get(i + 1) = d_dt2_psiR.get(i);
    }
    auto f = self_force_per_mass(d_psiR, u, charge, evolved_mass, imetric);
    auto dt_f =
        dt_self_force_per_mass(d_psiR, dt_d_psiR, u, dt_u_rollon, charge,
                               evolved_mass, imetric, dt_imetric);
    auto dt2_f = dt2_self_force_per_mass(
        d_psiR, dt_d_psiR, dt2_d_psiR, u, dt_u_rollon, dt2_u_rollon, charge,
        evolved_mass, imetric, dt_imetric, dt2_imetric);

    if (iteration > 1) {
      for (size_t i = 0; i < 4; ++i) {
        dt_f.get(i) += dt_mass_factor * f.get(i);
        dt2_f.get(i) +=
            2. * dt_mass_factor * dt_f.get(i) + dt2_mass_factor * f.get(i);
      }
    }
    const auto f_roll_on = tenex::evaluate<ti::A>(roll_on * f(ti::A));
    const auto dt_f_roll_on =
        tenex::evaluate<ti::A>(dt_roll_on * f(ti::A) + roll_on * dt_f(ti::A));
    const auto dt2_f_roll_on = tenex::evaluate<ti::A>(
        dt2_roll_on * f(ti::A) + 2. * dt_roll_on * dt_f(ti::A) +
        roll_on * dt2_f(ti::A));

    const auto cov_f =
        Du_self_force_per_mass(f_roll_on, dt_f_roll_on, u, christoffel);
    const auto dt_cov_f =
        dt_Du_self_force_per_mass(f_roll_on, dt_f_roll_on, dt2_f_roll_on, u,
                                  dt_u_rollon, christoffel, dt_christoffel);
    for (size_t i = 0; i < Dim; ++i) {
      get(*acceleration_terms)[i + 3] = f_roll_on.get(i);
      get(*acceleration_terms)[i + 6] = dt_f_roll_on.get(i);
      get(*acceleration_terms)[i + 9] = cov_f.get(i);
      get(*acceleration_terms)[i + 12] = dt_cov_f.get(i);
    }
  }
  for (size_t i = 0; i < Dim; ++i) {
    get(*acceleration_terms)[i] = acc.get(i);
  }
}
}  // namespace CurvedScalarWave::Worldtube
