// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Equations.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/Evolution/Systems/CurvedScalarWave/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
Scalar<DataVector> make_lapse(const DataVector& used_for_size) {
  return Scalar<DataVector>{make_with_value<DataVector>(used_for_size, 3.)};
}

template <size_t Dim>
tnsr::I<DataVector, Dim> make_shift(const DataVector& used_for_size) {
  auto shift = make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    shift.get(i) = make_with_value<DataVector>(used_for_size, i + 1.);
  }
  return shift;
}

template <size_t Dim>
tnsr::II<DataVector, Dim> make_inverse_spatial_metric(
    const DataVector& used_for_size) {
  auto metric =
      make_with_value<tnsr::II<DataVector, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      metric.get(i, j) =
          make_with_value<DataVector>(used_for_size, (i + 1.) * (j + 1.));
    }
  }
  return metric;
}

template <size_t Dim>
tnsr::i<DataVector, Dim> make_deriv_lapse(const DataVector& used_for_size) {
  auto deriv_lapse =
      make_with_value<tnsr::i<DataVector, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    deriv_lapse.get(i) =
        make_with_value<DataVector>(used_for_size, 2.5 * (i + 1.));
  }
  return deriv_lapse;
}

template <size_t Dim>
tnsr::iJ<DataVector, Dim> make_deriv_shift(const DataVector& used_for_size) {
  auto deriv_shift =
      make_with_value<tnsr::iJ<DataVector, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j < Dim; ++j) {
      deriv_shift.get(i, j) =
          make_with_value<DataVector>(used_for_size, 3. * (j + 1.) - i + 4.);
    }
  }
  return deriv_shift;
}

template <size_t Dim>
tnsr::I<DataVector, Dim> make_trace_spatial_christoffel_second_kind(
    const DataVector& used_for_size) {
  auto trace_christoffel =
      make_with_value<tnsr::I<DataVector, Dim>>(used_for_size, 0.);
  for (size_t i = 0; i < Dim; ++i) {
    trace_christoffel.get(i) =
        make_with_value<DataVector>(used_for_size, 3. * i - 2.5);
  }
  return trace_christoffel;
}

Scalar<DataVector> make_trace_extrinsic_curvature(
    const DataVector& used_for_size) {
  return Scalar<DataVector>{make_with_value<DataVector>(used_for_size, 5.)};
}

template <size_t Dim>
Variables<tmpl::list<CurvedScalarWave::Psi, CurvedScalarWave::Pi,
                     CurvedScalarWave::Phi<Dim>>>
calculate_du_dt(const DataVector& used_for_size) {
  auto dt_psi = make_with_value<Scalar<DataVector>>(used_for_size, 0.);
  auto dt_pi = make_with_value<Scalar<DataVector>>(used_for_size, 0.);
  auto dt_phi = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      used_for_size, 0.);
  CurvedScalarWave::ComputeDuDt<Dim>::apply(
      make_not_null(&dt_pi), make_not_null(&dt_phi), make_not_null(&dt_psi),
      make_pi(used_for_size), make_phi<Dim>(used_for_size),
      make_d_psi<Dim>(used_for_size), make_d_pi<Dim>(used_for_size),
      make_d_phi<Dim>(used_for_size), make_lapse(used_for_size),
      make_shift<Dim>(used_for_size), make_deriv_lapse<Dim>(used_for_size),
      make_deriv_shift<Dim>(used_for_size),
      make_inverse_spatial_metric<Dim>(used_for_size),
      make_trace_spatial_christoffel_second_kind<Dim>(used_for_size),
      make_trace_extrinsic_curvature(used_for_size),
      make_constraint_gamma1(used_for_size),
      make_constraint_gamma2(used_for_size));
  Variables<tmpl::list<CurvedScalarWave::Psi, CurvedScalarWave::Pi,
                       CurvedScalarWave::Phi<Dim>>>
      vars(used_for_size.size(), 0.);

  get<CurvedScalarWave::Psi>(vars) = dt_psi;
  get<CurvedScalarWave::Pi>(vars) = dt_pi;
  get<CurvedScalarWave::Phi<Dim>>(vars) = dt_phi;
  return vars;
}
template <size_t Dim>
void check_du_dt(const DataVector& used_for_size);

template <>
void check_du_dt<1>(const DataVector& used_for_size) {
  const auto vars = calculate_du_dt<1>(used_for_size);
  const auto& dt_psi = get<CurvedScalarWave::Psi>(vars);
  const auto& dt_pi = get<CurvedScalarWave::Pi>(vars);
  const auto& dt_phi = get<CurvedScalarWave::Phi<1>>(vars);
  CHECK_ITERABLE_APPROX(dt_psi.get(), (DataVector{used_for_size.size(), 5.3}));
  CHECK_ITERABLE_APPROX(dt_pi.get(), (DataVector{used_for_size.size(), -22.5}));
  CHECK_ITERABLE_APPROX(dt_phi.get(0), (DataVector{used_for_size.size(), 4.}));
}
template <>
void check_du_dt<2>(const DataVector& used_for_size) {
  const auto vars = calculate_du_dt<2>(used_for_size);
  const auto& dt_psi = get<CurvedScalarWave::Psi>(vars);
  const auto& dt_pi = get<CurvedScalarWave::Pi>(vars);
  const auto& dt_phi = get<CurvedScalarWave::Phi<2>>(vars);
  CHECK_ITERABLE_APPROX(dt_psi.get(), (DataVector{used_for_size.size(), 68.3}));
  CHECK_ITERABLE_APPROX(dt_pi.get(),
                        (DataVector{used_for_size.size(), -272.15}));
  CHECK_ITERABLE_APPROX(dt_phi.get(0),
                        (DataVector{used_for_size.size(), -15.}));
  CHECK_ITERABLE_APPROX(dt_phi.get(1),
                        (DataVector{used_for_size.size(), -78.}));
}
template <>
void check_du_dt<3>(const DataVector& used_for_size) {
  const auto vars = calculate_du_dt<3>(used_for_size);
  const auto& dt_psi = get<CurvedScalarWave::Psi>(vars);
  const auto& dt_pi = get<CurvedScalarWave::Pi>(vars);
  const auto& dt_phi = get<CurvedScalarWave::Phi<3>>(vars);
  CHECK_ITERABLE_APPROX(dt_psi.get(),
                        (DataVector{used_for_size.size(), 255.8}));
  CHECK_ITERABLE_APPROX(dt_pi.get(),
                        (DataVector{used_for_size.size(), -976.85}));
  CHECK_ITERABLE_APPROX(dt_phi.get(0),
                        (DataVector{used_for_size.size(), -62.5}));
  CHECK_ITERABLE_APPROX(dt_phi.get(1),
                        (DataVector{used_for_size.size(), -117.}));
  CHECK_ITERABLE_APPROX(dt_phi.get(2),
                        (DataVector{used_for_size.size(), -171.5}));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.CurvedScalarWave.DuDt",
                  "[Unit][Evolution]") {
  const DataVector used_for_size{2, 0.};
  check_du_dt<1>(used_for_size);
  check_du_dt<2>(used_for_size);
  check_du_dt<3>(used_for_size);
}
