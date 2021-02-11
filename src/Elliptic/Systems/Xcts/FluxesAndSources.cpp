// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Elasticity/Equations.hpp"
#include "Elliptic/Systems/Poisson/Equations.hpp"
#include "Elliptic/Systems/Xcts/Equations.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"

namespace Xcts {

void Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
  Poisson::flat_cartesian_fluxes(flux_for_conformal_factor,
                                 conformal_factor_gradient);
}

void Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor) noexcept {
  Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                            conformal_factor);
}

void Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient) noexcept {
  Poisson::curved_fluxes(flux_for_conformal_factor, inv_conformal_metric,
                         conformal_factor_gradient);
}

void Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_conformal_factor_gradient,
    const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
    const Scalar<DataVector>& conformal_factor) noexcept {
  Poisson::auxiliary_fluxes(flux_for_conformal_factor_gradient,
                            conformal_factor);
}

void Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3>&
        lapse_times_conformal_factor_gradient) noexcept {
  Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
      flux_for_conformal_factor, conformal_factor_gradient);
  Poisson::flat_cartesian_fluxes(flux_for_lapse_times_conformal_factor,
                                 lapse_times_conformal_factor_gradient);
}

void Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_conformal_factor_gradient,
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor_gradient,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  Fluxes<Equations::Hamiltonian, Geometry::FlatCartesian>::apply(
      flux_for_conformal_factor_gradient, conformal_factor);
  Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                            lapse_times_conformal_factor);
}

void Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3>&
        lapse_times_conformal_factor_gradient) noexcept {
  Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
      flux_for_conformal_factor, inv_conformal_metric,
      conformal_factor_gradient);
  Poisson::curved_fluxes(flux_for_lapse_times_conformal_factor,
                         inv_conformal_metric,
                         lapse_times_conformal_factor_gradient);
}

void Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_conformal_factor_gradient,
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor_gradient,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor) noexcept {
  Fluxes<Equations::Hamiltonian, Geometry::Curved>::apply(
      flux_for_conformal_factor_gradient, inv_conformal_metric,
      conformal_factor);
  Poisson::auxiliary_fluxes(flux_for_lapse_times_conformal_factor_gradient,
                            lapse_times_conformal_factor);
}

void Fluxes<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian>::
    apply(
        const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
        const gsl::not_null<tnsr::I<DataVector, 3>*>
            flux_for_lapse_times_conformal_factor,
        const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
        const tnsr::i<DataVector, 3>& conformal_factor_gradient,
        const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
        const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
  Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
      flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
      conformal_factor_gradient, lapse_times_conformal_factor_gradient);
  Xcts::longitudinal_operator_flat_cartesian(longitudinal_shift_excess,
                                             shift_strain);
}

void Fluxes<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian>::
    apply(const gsl::not_null<tnsr::Ij<DataVector, 3>*>
              flux_for_conformal_factor_gradient,
          const gsl::not_null<tnsr::Ij<DataVector, 3>*>
              flux_for_lapse_times_conformal_factor_gradient,
          const gsl::not_null<tnsr::Ijj<DataVector, 3>*> flux_for_shift_strain,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& shift) noexcept {
  Fluxes<Equations::HamiltonianAndLapse, Geometry::FlatCartesian>::apply(
      flux_for_conformal_factor_gradient,
      flux_for_lapse_times_conformal_factor_gradient, conformal_factor,
      lapse_times_conformal_factor);
  Elasticity::auxiliary_fluxes(flux_for_shift_strain, shift);
}

void Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> flux_for_conformal_factor,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor,
    const gsl::not_null<tnsr::II<DataVector, 3>*> longitudinal_shift_excess,
    const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::i<DataVector, 3>& conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& lapse_times_conformal_factor_gradient,
    const tnsr::ii<DataVector, 3>& shift_strain) noexcept {
  Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
      flux_for_conformal_factor, flux_for_lapse_times_conformal_factor,
      inv_conformal_metric, conformal_factor_gradient,
      lapse_times_conformal_factor_gradient);
  Xcts::longitudinal_operator(longitudinal_shift_excess, shift_strain,
                              inv_conformal_metric);
}

void Fluxes<Equations::HamiltonianLapseAndShift, Geometry::Curved>::apply(
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_conformal_factor_gradient,
    const gsl::not_null<tnsr::Ij<DataVector, 3>*>
        flux_for_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::Ijj<DataVector, 3>*> flux_for_shift_strain,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const Scalar<DataVector>& conformal_factor,
    const Scalar<DataVector>& lapse_times_conformal_factor,
    const tnsr::I<DataVector, 3>& shift_excess) noexcept {
  Fluxes<Equations::HamiltonianAndLapse, Geometry::Curved>::apply(
      flux_for_conformal_factor_gradient,
      flux_for_lapse_times_conformal_factor_gradient, inv_conformal_metric,
      conformal_factor, lapse_times_conformal_factor);
  Elasticity::curved_auxiliary_fluxes(flux_for_shift_strain, conformal_metric,
                                      shift_excess);
}

template <int ConformalMatterScale>
void Sources<Equations::Hamiltonian, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
          const Scalar<DataVector>& conformal_factor,
          const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/) noexcept {
  add_hamiltonian_sources<ConformalMatterScale>(
      hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor);
  add_distortion_hamiltonian_sources(
      hamiltonian_constraint,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor);
}

template <int ConformalMatterScale>
void Sources<Equations::Hamiltonian, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>&
          /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
          const Scalar<DataVector>& /*conformal_factor*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void Sources<Equations::Hamiltonian, Geometry::Curved, ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const tnsr::I<DataVector, 3>& conformal_factor_flux) noexcept {
  Sources<Equations::Hamiltonian, Geometry::FlatCartesian,
          ConformalMatterScale>::
      apply(hamiltonian_constraint, conformal_energy_density,
            extrinsic_curvature_trace,
            longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
            conformal_factor, conformal_factor_flux);
  add_curved_hamiltonian_or_lapse_sources(
      hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
  Poisson::add_curved_sources(hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux);
}

template <int ConformalMatterScale>
void Sources<Equations::Hamiltonian, Geometry::Curved, ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>&
          /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
          const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
          const Scalar<DataVector>& /*conformal_ricci_scalar*/,
          const Scalar<DataVector>& /*conformal_factor*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources and the covariant
  // derivative of a scalar is just a partial derivative, so no
  // Christoffel-symbol terms need to be added here.
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> lapse_equation,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_square,
          const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
          const tnsr::I<DataVector, 3>&
          /*lapse_times_conformal_factor_flux*/) noexcept {
  add_hamiltonian_sources<ConformalMatterScale>(
      hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor);
  add_lapse_sources<ConformalMatterScale>(
      lapse_equation, conformal_energy_density, conformal_stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
      lapse_times_conformal_factor);
  add_distortion_hamiltonian_and_lapse_sources(
      hamiltonian_constraint, lapse_equation,
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor);
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_conformal_factor_gradient*/,
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_lapse_times_conformal_factor_gradient*/,
        const Scalar<DataVector>& /*conformal_energy_density*/,
        const Scalar<DataVector>& /*conformal_stress_trace*/,
        const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
        const Scalar<DataVector>&
        /*longitudinal_shift_minus_dt_conformal_metric_square*/,
        const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*conformal_factor*/,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianAndLapse, Geometry::Curved,
             ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> lapse_equation,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_square,
          const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& conformal_factor_flux,
          const tnsr::I<DataVector, 3>&
              lapse_times_conformal_factor_flux) noexcept {
  Sources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
          ConformalMatterScale>::
      apply(hamiltonian_constraint, lapse_equation, conformal_energy_density,
            conformal_stress_trace, extrinsic_curvature_trace,
            dt_extrinsic_curvature_trace,
            longitudinal_shift_minus_dt_conformal_metric_square,
            shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
            lapse_times_conformal_factor, conformal_factor_flux,
            lapse_times_conformal_factor_flux);
  add_curved_hamiltonian_or_lapse_sources(
      hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
  Poisson::add_curved_sources(hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux);
  add_curved_hamiltonian_or_lapse_sources(
      lapse_equation, conformal_ricci_scalar, lapse_times_conformal_factor);
  Poisson::add_curved_sources(lapse_equation, conformal_christoffel_contracted,
                              lapse_times_conformal_factor_flux);
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianAndLapse, Geometry::Curved,
             ConformalMatterScale>::
    apply(
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_conformal_factor_gradient*/,
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_lapse_times_conformal_factor_gradient*/,
        const Scalar<DataVector>& /*conformal_energy_density*/,
        const Scalar<DataVector>& /*conformal_stress_trace*/,
        const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
        const Scalar<DataVector>&
        /*longitudinal_shift_minus_dt_conformal_metric_square*/,
        const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
        const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
        const Scalar<DataVector>& /*conformal_ricci_scalar*/,
        const Scalar<DataVector>& /*conformal_factor*/,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources and the covariant
  // derivative of a scalar is just a partial derivative, so no
  // Christoffel-symbol terms need to be added here.
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> lapse_equation,
          const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const tnsr::I<DataVector, 3>& conformal_momentum_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
          const tnsr::I<DataVector, 3>& shift_background,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_background_minus_dt_conformal_metric,
          const tnsr::I<DataVector, 3>&
              div_longitudinal_shift_background_minus_dt_conformal_metric,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& shift_excess,
          const tnsr::I<DataVector, 3>& conformal_factor_flux,
          const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
          const tnsr::II<DataVector, 3>& longitudinal_shift_excess) noexcept {
  auto shift = shift_background;
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) += shift_excess.get(i);
  }
  auto longitudinal_shift_minus_dt_conformal_metric =
      longitudinal_shift_background_minus_dt_conformal_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
          longitudinal_shift_excess.get(i, j);
    }
  }
  add_hamiltonian_sources<ConformalMatterScale>(
      hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor);
  add_lapse_sources<ConformalMatterScale>(
      lapse_equation, conformal_energy_density, conformal_stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor);
  add_flat_cartesian_momentum_sources<ConformalMatterScale>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      conformal_momentum_density, extrinsic_curvature_trace_gradient,
      div_longitudinal_shift_background_minus_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianLapseAndShift, Geometry::FlatCartesian,
             ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient*/,
          const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_lapse_times_conformal_factor_gradient*/,
          const gsl::not_null<tnsr::ii<DataVector, 3>*>
          /*equation_for_shift_strain*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*conformal_stress_trace*/,
          const tnsr::I<DataVector, 3>& /*conformal_momentum_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
          const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
          const tnsr::I<DataVector, 3>& /*shift_background*/,
          const tnsr::II<DataVector, 3>&
          /*longitudinal_shift_background_minus_dt_conformal_metric*/,
          const tnsr::I<DataVector, 3>&
          /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
          const Scalar<DataVector>& /*conformal_factor*/,
          const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
          const tnsr::I<DataVector, 3>& /*shift_excess*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianLapseAndShift, Geometry::Curved,
             ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*> hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> lapse_equation,
          const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const tnsr::I<DataVector, 3>& conformal_momentum_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
          const tnsr::I<DataVector, 3>& shift_background,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_background_minus_dt_conformal_metric,
          const tnsr::I<DataVector, 3>&
              div_longitudinal_shift_background_minus_dt_conformal_metric,
          const tnsr::ii<DataVector, 3>& conformal_metric,
          const tnsr::II<DataVector, 3>& inv_conformal_metric,
          const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
          const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& shift_excess,
          const tnsr::I<DataVector, 3>& conformal_factor_flux,
          const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
          const tnsr::II<DataVector, 3>& longitudinal_shift_excess) noexcept {
  auto shift = shift_background;
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) += shift_excess.get(i);
  }
  auto longitudinal_shift_minus_dt_conformal_metric =
      longitudinal_shift_background_minus_dt_conformal_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
          longitudinal_shift_excess.get(i, j);
    }
  }
  add_hamiltonian_sources<ConformalMatterScale>(
      hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor);
  add_curved_hamiltonian_or_lapse_sources(
      hamiltonian_constraint, conformal_ricci_scalar, conformal_factor);
  Poisson::add_curved_sources(hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux);
  add_lapse_sources<ConformalMatterScale>(
      lapse_equation, conformal_energy_density, conformal_stress_trace,
      extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor);
  add_curved_hamiltonian_or_lapse_sources(
      lapse_equation, conformal_ricci_scalar, lapse_times_conformal_factor);
  Poisson::add_curved_sources(lapse_equation, conformal_christoffel_contracted,
                              lapse_times_conformal_factor_flux);
  add_curved_momentum_sources<ConformalMatterScale>(
      hamiltonian_constraint, lapse_equation, momentum_constraint,
      conformal_momentum_density, extrinsic_curvature_trace_gradient,
      conformal_metric, inv_conformal_metric,
      div_longitudinal_shift_background_minus_dt_conformal_metric,
      conformal_factor, lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric);
  Elasticity::add_curved_sources(momentum_constraint,
                                 conformal_christoffel_second_kind,
                                 conformal_christoffel_contracted,
                                 longitudinal_shift_minus_dt_conformal_metric);
}

template <int ConformalMatterScale>
void Sources<Equations::HamiltonianLapseAndShift, Geometry::Curved,
             ConformalMatterScale>::
    apply(
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_conformal_factor_gradient*/,
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_lapse_times_conformal_factor_gradient*/,
        const gsl::not_null<tnsr::ii<DataVector, 3>*> equation_for_shift_strain,
        const Scalar<DataVector>& /*conformal_energy_density*/,
        const Scalar<DataVector>& /*conformal_stress_trace*/,
        const tnsr::I<DataVector, 3>& /*conformal_momentum_density*/,
        const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
        const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
        const tnsr::I<DataVector, 3>& /*shift_background*/,
        const tnsr::II<DataVector, 3>&
        /*longitudinal_shift_background_minus_dt_conformal_metric*/,
        const tnsr::I<DataVector, 3>&
        /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
        const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
        const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
        const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
        const tnsr::Ijj<DataVector, 3>& /*conformal_christoffel_second_kind*/,
        const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
        const Scalar<DataVector>& /*conformal_ricci_scalar*/,
        const Scalar<DataVector>& /*conformal_factor*/,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
        const tnsr::I<DataVector, 3>& shift_excess) noexcept {
  Elasticity::add_curved_auxiliary_sources(equation_for_shift_strain,
                                           conformal_christoffel_first_kind,
                                           shift_excess);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::Hamiltonian, Geometry::FlatCartesian,
                       ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& conformal_factor_correction,
          const tnsr::I<DataVector, 3>&
          /*conformal_factor_flux_correction*/) noexcept {
  add_linearized_hamiltonian_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor, conformal_factor_correction);
  add_linearized_distortion_hamiltonian_sources(
      linearized_hamiltonian_constraint,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      conformal_factor, conformal_factor_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::Hamiltonian, Geometry::FlatCartesian,
                       ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*source_for_conformal_factor_gradient_correction*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>&
          /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
          const Scalar<DataVector>& /*conformal_factor*/,
          const Scalar<DataVector>& /*conformal_factor_correction*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::Hamiltonian, Geometry::Curved,
                       ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& conformal_factor_correction,
          const tnsr::I<DataVector, 3>&
              conformal_factor_flux_correction) noexcept {
  LinearizedSources<Equations::Hamiltonian, Geometry::FlatCartesian,
                    ConformalMatterScale>::
      apply(linearized_hamiltonian_constraint, conformal_energy_density,
            extrinsic_curvature_trace,
            longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
            conformal_factor, conformal_factor_correction,
            conformal_factor_flux_correction);
  add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                          conformal_ricci_scalar,
                                          conformal_factor_correction);
  Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::Hamiltonian, Geometry::Curved,
                       ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient_correction*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>&
          /*longitudinal_shift_minus_dt_conformal_metric_over_lapse_square*/,
          const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
          const Scalar<DataVector>& /*conformal_ricci_scalar*/,
          const Scalar<DataVector>& /*conformal_factor*/,
          const Scalar<DataVector>& /*conformal_factor_correction*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources and the covariant
  // derivative of a scalar is just a partial derivative, so no
  // Christoffel-symbol terms need to be added here.
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
                       ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_square,
          const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const Scalar<DataVector>& conformal_factor_correction,
          const Scalar<DataVector>& lapse_times_conformal_factor_correction,
          const tnsr::I<DataVector, 3>& /*conformal_factor_flux_correction*/,
          const tnsr::I<DataVector, 3>&
          /*lapse_times_conformal_factor_flux_correction*/) noexcept {
  add_linearized_hamiltonian_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor, conformal_factor_correction);
  add_linearized_lapse_sources<ConformalMatterScale>(
      linearized_lapse_equation, conformal_energy_density,
      conformal_stress_trace, extrinsic_curvature_trace,
      dt_extrinsic_curvature_trace, shift_dot_deriv_extrinsic_curvature_trace,
      conformal_factor, lapse_times_conformal_factor,
      conformal_factor_correction, lapse_times_conformal_factor_correction);
  add_linearized_distortion_hamiltonian_and_lapse_sources(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      longitudinal_shift_minus_dt_conformal_metric_square, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
                       ConformalMatterScale>::
    apply(
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_conformal_factor_gradient_correction*/,
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
        const Scalar<DataVector>& /*conformal_energy_density*/,
        const Scalar<DataVector>& /*conformal_stress_trace*/,
        const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
        const Scalar<DataVector>&
        /*longitudinal_shift_minus_dt_conformal_metric_square*/,
        const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*conformal_factor*/,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
        const Scalar<DataVector>& /*conformal_factor_correction*/,
        const Scalar<
            DataVector>& /*lapse_times_conformal_factor_correction*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianAndLapse, Geometry::Curved,
                       ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const Scalar<DataVector>&
              longitudinal_shift_minus_dt_conformal_metric_square,
          const Scalar<DataVector>& shift_dot_deriv_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const Scalar<DataVector>& conformal_factor_correction,
          const Scalar<DataVector>& lapse_times_conformal_factor_correction,
          const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
          const tnsr::I<DataVector, 3>&
              lapse_times_conformal_factor_flux_correction) noexcept {
  LinearizedSources<Equations::HamiltonianAndLapse, Geometry::FlatCartesian,
                    ConformalMatterScale>::
      apply(linearized_hamiltonian_constraint, linearized_lapse_equation,
            conformal_energy_density, conformal_stress_trace,
            extrinsic_curvature_trace, dt_extrinsic_curvature_trace,
            longitudinal_shift_minus_dt_conformal_metric_square,
            shift_dot_deriv_extrinsic_curvature_trace, conformal_factor,
            lapse_times_conformal_factor, conformal_factor_correction,
            lapse_times_conformal_factor_correction,
            conformal_factor_flux_correction,
            lapse_times_conformal_factor_flux_correction);
  add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                          conformal_ricci_scalar,
                                          conformal_factor_correction);
  Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux_correction);
  add_curved_hamiltonian_or_lapse_sources(
      linearized_lapse_equation, conformal_ricci_scalar,
      lapse_times_conformal_factor_correction);
  Poisson::add_curved_sources(linearized_lapse_equation,
                              conformal_christoffel_contracted,
                              lapse_times_conformal_factor_flux_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianAndLapse, Geometry::Curved,
                       ConformalMatterScale>::
    apply(
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_conformal_factor_gradient_correction*/,
        const gsl::not_null<tnsr::i<DataVector, 3>*>
        /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
        const Scalar<DataVector>& /*conformal_energy_density*/,
        const Scalar<DataVector>& /*conformal_stress_trace*/,
        const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
        const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
        const Scalar<DataVector>&
        /*longitudinal_shift_minus_dt_conformal_metric_square*/,
        const Scalar<DataVector>& /*shift_dot_deriv_extrinsic_curvature_trace*/,
        const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
        const Scalar<DataVector>& /*conformal_ricci_scalar*/,
        const Scalar<DataVector>& /*conformal_factor*/,
        const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
        const Scalar<DataVector>& /*conformal_factor_correction*/,
        const Scalar<
            DataVector>& /*lapse_times_conformal_factor_correction*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources and the covariant
  // derivative of a scalar is just a partial derivative, so no
  // Christoffel-symbol terms need to be added here.
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianLapseAndShift,
                       Geometry::FlatCartesian, ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
          const gsl::not_null<tnsr::I<DataVector, 3>*>
              linearized_momentum_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const tnsr::I<DataVector, 3>& conformal_momentum_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
          const tnsr::I<DataVector, 3>& shift_background,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_background_minus_dt_conformal_metric,
          const tnsr::I<DataVector, 3>&
          /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& shift_excess,
          const tnsr::I<DataVector, 3>& conformal_factor_flux,
          const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
          const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
          const Scalar<DataVector>& conformal_factor_correction,
          const Scalar<DataVector>& lapse_times_conformal_factor_correction,
          const tnsr::I<DataVector, 3>& shift_excess_correction,
          const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
          const tnsr::I<DataVector, 3>&
              lapse_times_conformal_factor_flux_correction,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_excess_correction) noexcept {
  auto shift = shift_background;
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) += shift_excess.get(i);
  }
  auto longitudinal_shift_minus_dt_conformal_metric =
      longitudinal_shift_background_minus_dt_conformal_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
          longitudinal_shift_excess.get(i, j);
    }
  }
  add_linearized_hamiltonian_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor, conformal_factor_correction);
  add_linearized_lapse_sources<ConformalMatterScale>(
      linearized_lapse_equation, conformal_energy_density,
      conformal_stress_trace, extrinsic_curvature_trace,
      dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  add_flat_cartesian_linearized_momentum_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, conformal_momentum_density,
      extrinsic_curvature_trace_gradient, conformal_factor,
      lapse_times_conformal_factor, conformal_factor_flux,
      lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_excess_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_excess_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianLapseAndShift,
                       Geometry::FlatCartesian, ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient_correction*/,
          const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
          const gsl::not_null<tnsr::ii<DataVector, 3>*>
          /*equation_for_shift_strain_correction*/,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*conformal_stress_trace*/,
          const tnsr::I<DataVector, 3>& /*conformal_momentum_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
          const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
          const tnsr::I<DataVector, 3>& /*shift_background*/,
          const tnsr::II<DataVector, 3>&
          /*longitudinal_shift_background_minus_dt_conformal_metric*/,
          const tnsr::I<DataVector, 3>&
          /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
          const Scalar<DataVector>& /*conformal_factor*/,
          const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
          const tnsr::I<DataVector, 3>& /*shift_excess*/,
          const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
          const tnsr::I<DataVector, 3>& /*lapse_times_conformal_factor_flux*/,
          const tnsr::II<DataVector, 3>& /*longitudinal_shift_excess*/,
          const Scalar<DataVector>& /*conformal_factor_correction*/,
          const Scalar<DataVector>& /*lapse_times_conformal_factor_correction*/,
          const tnsr::I<DataVector, 3>& /*shift_excess_correction*/) noexcept {
  // Nothing to do. The auxiliary equation has no sources.
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianLapseAndShift, Geometry::Curved,
                       ConformalMatterScale>::
    apply(const gsl::not_null<Scalar<DataVector>*>
              linearized_hamiltonian_constraint,
          const gsl::not_null<Scalar<DataVector>*> linearized_lapse_equation,
          const gsl::not_null<tnsr::I<DataVector, 3>*>
              linearized_momentum_constraint,
          const Scalar<DataVector>& conformal_energy_density,
          const Scalar<DataVector>& conformal_stress_trace,
          const tnsr::I<DataVector, 3>& conformal_momentum_density,
          const Scalar<DataVector>& extrinsic_curvature_trace,
          const Scalar<DataVector>& dt_extrinsic_curvature_trace,
          const tnsr::i<DataVector, 3>& extrinsic_curvature_trace_gradient,
          const tnsr::I<DataVector, 3>& shift_background,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_background_minus_dt_conformal_metric,
          const tnsr::I<DataVector, 3>&
          /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
          const tnsr::ii<DataVector, 3>& conformal_metric,
          const tnsr::II<DataVector, 3>& inv_conformal_metric,
          const tnsr::ijj<DataVector, 3>& /*conformal_christoffel_first_kind*/,
          const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
          const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
          const Scalar<DataVector>& conformal_ricci_scalar,
          const Scalar<DataVector>& conformal_factor,
          const Scalar<DataVector>& lapse_times_conformal_factor,
          const tnsr::I<DataVector, 3>& shift_excess,
          const tnsr::I<DataVector, 3>& conformal_factor_flux,
          const tnsr::I<DataVector, 3>& lapse_times_conformal_factor_flux,
          const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
          const Scalar<DataVector>& conformal_factor_correction,
          const Scalar<DataVector>& lapse_times_conformal_factor_correction,
          const tnsr::I<DataVector, 3>& shift_excess_correction,
          const tnsr::I<DataVector, 3>& conformal_factor_flux_correction,
          const tnsr::I<DataVector, 3>&
              lapse_times_conformal_factor_flux_correction,
          const tnsr::II<DataVector, 3>&
              longitudinal_shift_excess_correction) noexcept {
  auto shift = shift_background;
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) += shift_excess.get(i);
  }
  auto longitudinal_shift_minus_dt_conformal_metric =
      longitudinal_shift_background_minus_dt_conformal_metric;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      longitudinal_shift_minus_dt_conformal_metric.get(i, j) +=
          longitudinal_shift_excess.get(i, j);
    }
  }
  add_linearized_hamiltonian_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, conformal_energy_density,
      extrinsic_curvature_trace, conformal_factor, conformal_factor_correction);
  add_curved_hamiltonian_or_lapse_sources(linearized_hamiltonian_constraint,
                                          conformal_ricci_scalar,
                                          conformal_factor_correction);
  Poisson::add_curved_sources(linearized_hamiltonian_constraint,
                              conformal_christoffel_contracted,
                              conformal_factor_flux_correction);
  add_linearized_lapse_sources<ConformalMatterScale>(
      linearized_lapse_equation, conformal_energy_density,
      conformal_stress_trace, extrinsic_curvature_trace,
      dt_extrinsic_curvature_trace,
      dot_product(shift, extrinsic_curvature_trace_gradient), conformal_factor,
      lapse_times_conformal_factor, conformal_factor_correction,
      lapse_times_conformal_factor_correction);
  add_curved_hamiltonian_or_lapse_sources(
      linearized_lapse_equation, conformal_ricci_scalar,
      lapse_times_conformal_factor_correction);
  Poisson::add_curved_sources(linearized_lapse_equation,
                              conformal_christoffel_contracted,
                              lapse_times_conformal_factor_flux_correction);
  add_curved_linearized_momentum_sources<ConformalMatterScale>(
      linearized_hamiltonian_constraint, linearized_lapse_equation,
      linearized_momentum_constraint, conformal_momentum_density,
      extrinsic_curvature_trace_gradient, conformal_metric,
      inv_conformal_metric, conformal_factor, lapse_times_conformal_factor,
      conformal_factor_flux, lapse_times_conformal_factor_flux,
      longitudinal_shift_minus_dt_conformal_metric, conformal_factor_correction,
      lapse_times_conformal_factor_correction, shift_excess_correction,
      conformal_factor_flux_correction,
      lapse_times_conformal_factor_flux_correction,
      longitudinal_shift_excess_correction);
  Elasticity::add_curved_sources(
      linearized_momentum_constraint, conformal_christoffel_second_kind,
      conformal_christoffel_contracted, longitudinal_shift_excess_correction);
}

template <int ConformalMatterScale>
void LinearizedSources<Equations::HamiltonianLapseAndShift, Geometry::Curved,
                       ConformalMatterScale>::
    apply(const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_conformal_factor_gradient_correction*/,
          const gsl::not_null<tnsr::i<DataVector, 3>*>
          /*equation_for_lapse_times_conformal_factor_gradient_correction*/,
          const gsl::not_null<tnsr::ii<DataVector, 3>*>
              equation_for_shift_strain_correction,
          const Scalar<DataVector>& /*conformal_energy_density*/,
          const Scalar<DataVector>& /*conformal_stress_trace*/,
          const tnsr::I<DataVector, 3>& /*conformal_momentum_density*/,
          const Scalar<DataVector>& /*extrinsic_curvature_trace*/,
          const Scalar<DataVector>& /*dt_extrinsic_curvature_trace*/,
          const tnsr::i<DataVector, 3>& /*extrinsic_curvature_trace_gradient*/,
          const tnsr::I<DataVector, 3>& /*shift_background*/,
          const tnsr::II<DataVector, 3>&
          /*longitudinal_shift_background_minus_dt_conformal_metric*/,
          const tnsr::I<DataVector, 3>&
          /*div_longitudinal_shift_background_minus_dt_conformal_metric*/,
          const tnsr::ii<DataVector, 3>& /*conformal_metric*/,
          const tnsr::II<DataVector, 3>& /*inv_conformal_metric*/,
          const tnsr::ijj<DataVector, 3>& conformal_christoffel_first_kind,
          const tnsr::Ijj<DataVector, 3>& /*conformal_christoffel_second_kind*/,
          const tnsr::i<DataVector, 3>& /*conformal_christoffel_contracted*/,
          const Scalar<DataVector>& /*conformal_ricci_scalar*/,
          const Scalar<DataVector>& /*conformal_factor*/,
          const Scalar<DataVector>& /*lapse_times_conformal_factor*/,
          const tnsr::I<DataVector, 3>& /*shift_excess*/,
          const tnsr::I<DataVector, 3>& /*conformal_factor_flux*/,
          const tnsr::I<DataVector, 3>& /*lapse_times_conformal_factor_flux*/,
          const tnsr::II<DataVector, 3>& /*longitudinal_shift_excess*/,
          const Scalar<DataVector>& /*conformal_factor_correction*/,
          const Scalar<DataVector>& /*lapse_times_conformal_factor_correction*/,
          const tnsr::I<DataVector, 3>& shift_excess_correction) noexcept {
  Elasticity::add_curved_auxiliary_sources(equation_for_shift_strain_correction,
                                           conformal_christoffel_first_kind,
                                           shift_excess_correction);
}

#define EQNS(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GEOM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define CONF_MATTER_SCALE(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATION(r, data)                                             \
  template class Sources<EQNS(data), GEOM(data), CONF_MATTER_SCALE(data)>; \
  template class LinearizedSources<EQNS(data), GEOM(data),                 \
                                   CONF_MATTER_SCALE(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION,
                        (Equations::Hamiltonian, Equations::HamiltonianAndLapse,
                         Equations::HamiltonianLapseAndShift),
                        (Geometry::FlatCartesian, Geometry::Curved), (0, 6, 8))

#undef EQNS
#undef GEOM
#undef CONF_MATTER_SCALE
#undef INSTANTIATION

}  // namespace Xcts
