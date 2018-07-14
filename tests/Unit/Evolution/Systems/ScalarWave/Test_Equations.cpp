// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"  // IWYU pragma: keep
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Dim>
void check_du_dt(const size_t npts, const double time) {
  ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian>(1.0, 1.0, 0.0));

  tnsr::I<DataVector, Dim> x = [npts]() {
    auto logical_coords = logical_coordinates(Mesh<Dim>{
        3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  auto box = db::create<db::AddSimpleTags<
      Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
      Tags::dt<ScalarWave::Psi>, ScalarWave::Pi,
      Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
      Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(
      Scalar<DataVector>(pow<Dim>(npts), 0.0),
      tnsr::i<DataVector, Dim, Frame::Inertial>(pow<Dim>(npts), 0.0),
      Scalar<DataVector>(pow<Dim>(npts), 0.0),
      Scalar<DataVector>(-1.0 * solution.dpsi_dt(x, time).get()),
      [&x, &time, &solution ]() noexcept {
        auto dpi_dx = solution.d2psi_dtdx(x, time);
        for (size_t i = 0; i < Dim; ++i) {
          dpi_dx.get(i) *= -1.0;
        }
        return dpi_dx;
      }(),
      [&npts, &x, &time, &solution ]() noexcept {
        tnsr::ij<DataVector, Dim, Frame::Inertial> d2psi_dxdx{pow<Dim>(npts),
                                                              0.0};
        const tnsr::ii<DataVector, Dim, Frame::Inertial> ddpsi_soln =
            solution.d2psi_dxdx(x, time);
        for (size_t i = 0; i < Dim; ++i) {
          for (size_t j = 0; j < Dim; ++j) {
            d2psi_dxdx.get(i, j) = ddpsi_soln.get(i, j);
          }
        }
        return d2psi_dxdx;
      }());

  db::mutate_apply<
      tmpl::list<Tags::dt<ScalarWave::Pi>, Tags::dt<ScalarWave::Phi<Dim>>,
                 Tags::dt<ScalarWave::Psi>>,
      typename ScalarWave::ComputeDuDt<Dim>::argument_tags>(
      ScalarWave::ComputeDuDt<Dim>{}, make_not_null(&box));

  CHECK_ITERABLE_APPROX(
      db::get<Tags::dt<ScalarWave::Pi>>(box),
      Scalar<DataVector>(-1.0 * solution.d2psi_dt2(x, time).get()));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Phi<Dim>>>(box),
                        solution.d2psi_dtdx(x, time));
  CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Psi>>(box),
                        solution.dpsi_dt(x, time));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.DuDt",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_du_dt<1>(3, time);
  check_du_dt<2>(3, time);
  check_du_dt<3>(3, time);
}

namespace {
template <size_t Dim>
void check_normal_dot_fluxes(const size_t npts, const double t) {
  const ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian>(1.0, 1.0, 0.0));

  const tnsr::I<DataVector, Dim> x = [npts]() {
    auto logical_coords = logical_coordinates(Mesh<Dim>{
        3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  const auto psi = solution.psi(x, t);
  const auto pi = Scalar<DataVector>{-get(solution.dpsi_dt(x, t))};
  const auto phi = solution.dpsi_dx(x, t);

  // Any numbers are fine, doesn't have anything to do with unit normal
  auto unit_normal =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(pi, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    unit_normal.get(d) = x.get(d);
  }

  auto normal_dot_flux_pi = make_with_value<Scalar<DataVector>>(pi, 0.0);
  auto normal_dot_flux_psi = make_with_value<Scalar<DataVector>>(pi, 0.0);
  auto normal_dot_flux_phi =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(pi, 0.0);

  ScalarWave::ComputeNormalDotFluxes<Dim>::apply(
      make_not_null(&normal_dot_flux_pi), make_not_null(&normal_dot_flux_phi),
      make_not_null(&normal_dot_flux_psi), pi, phi, unit_normal);

  CHECK(get(normal_dot_flux_psi) == DataVector(pow<Dim>(npts), 0.0));

  DataVector normal_dot_flux_pi_expected(pow<Dim>(npts), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_dot_flux_pi_expected += unit_normal.get(d) * phi.get(d);
  }
  CHECK(get(normal_dot_flux_pi) == normal_dot_flux_pi_expected);

  auto normal_dot_flux_phi_expected =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(pi, 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_dot_flux_phi_expected.get(d) = unit_normal.get(d) * get(pi);
  }
  CHECK(normal_dot_flux_phi == normal_dot_flux_phi_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.NormalDotFluxes",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_normal_dot_fluxes<1>(3, time);
  check_normal_dot_fluxes<2>(3, time);
  check_normal_dot_fluxes<3>(3, time);
}

namespace {
template <class... Tags, class FluxType, class... NormalDotNumericalFluxTypes>
void apply_numerical_flux(
    const FluxType& flux,
    const Variables<tmpl::list<Tags...>>& packaged_data_int,
    const Variables<tmpl::list<Tags...>>& packaged_data_ext,
    NormalDotNumericalFluxTypes&&... normal_dot_numerical_flux) {
  flux(std::forward<NormalDotNumericalFluxTypes>(normal_dot_numerical_flux)...,
       get<Tags>(packaged_data_int)..., get<Tags>(packaged_data_ext)...);
}

template <size_t Dim>
void check_upwind_flux(const size_t npts, const double t) {
  const ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian>(1.0, 1.0, 0.0));

  const tnsr::I<DataVector, Dim> x = [npts]() {
    auto logical_coords = logical_coordinates(Mesh<Dim>{
        3, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto});
    tnsr::I<DataVector, Dim> coords{pow<Dim>(npts)};
    for (size_t i = 0; i < Dim; ++i) {
      coords.get(i) = std::move(logical_coords.get(i));
    }
    return coords;
  }();

  // Any numbers are fine, doesn't have anything to do with unit normal
  tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal(pow<Dim>(npts), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    unit_normal.get(d) = x.get(d);
  }

  Variables<typename ScalarWave::UpwindFlux<Dim>::package_tags>
      packaged_data_int(pow<Dim>(npts), 0.0);
  Variables<typename ScalarWave::UpwindFlux<Dim>::package_tags>
      packaged_data_ext(pow<Dim>(npts), 0.0);

  ScalarWave::UpwindFlux<Dim> flux_computer{};
  flux_computer.package_data(
      make_not_null(&packaged_data_int), solution.dpsi_dt(x, t + 1.0),
      solution.dpsi_dx(x, t + 2.0),
      solution.psi(x, t + 4.0), unit_normal);
  flux_computer.package_data(
      make_not_null(&packaged_data_ext), solution.dpsi_dt(x, 2.0 * t + 10.0),
      solution.dpsi_dx(x, 2.0 * t + 9.0),
      solution.psi(x, 2.0 * t + 7.0), unit_normal);

  Scalar<DataVector> normal_dot_numerical_flux_pi(pow<Dim>(npts), 0.0);
  Scalar<DataVector> normal_dot_numerical_flux_psi(pow<Dim>(npts), 0.0);
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_dot_numerical_flux_phi(
      pow<Dim>(npts), 0.0);
  apply_numerical_flux(flux_computer, packaged_data_int, packaged_data_ext,
                       make_not_null(&normal_dot_numerical_flux_pi),
                       make_not_null(&normal_dot_numerical_flux_phi),
                       make_not_null(&normal_dot_numerical_flux_psi));

  CHECK(normal_dot_numerical_flux_psi ==
        Scalar<DataVector>(pow<Dim>(npts), 0.0));
  CHECK(normal_dot_numerical_flux_pi ==
        Scalar<DataVector>(0.5 * (get(solution.psi(x, t + 4.0)) -
                                  get(solution.psi(x, 2.0 * t + 7.0)) +
                                  get(solution.dpsi_dt(x, t + 1.0)) -
                                  get(solution.dpsi_dt(x, 2.0 * t + 10.0)))));

  tnsr::i<DataVector, Dim> normal_dot_numerical_flux_phi_expected(
      pow<Dim>(npts), 0.0);
  for (size_t d = 0; d < Dim; ++d) {
    normal_dot_numerical_flux_phi_expected.get(d) =
        0.5 * (solution.dpsi_dx(x, t + 2.0).get(d) -
               solution.dpsi_dx(x, 2.0 * t + 9.0).get(d) +
               unit_normal.get(d) * solution.dpsi_dt(x, t + 1.0).get() -
               unit_normal.get(d) * solution.dpsi_dt(x, 2.0 * t + 10.0).get());
  }
  CHECK(normal_dot_numerical_flux_phi ==
        normal_dot_numerical_flux_phi_expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.UpwindFlux",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  check_upwind_flux<1>(3, time);
  check_upwind_flux<2>(3, time);
  check_upwind_flux<3>(3, time);
}

static_assert(1.0 == ScalarWave::ComputeLargestCharacteristicSpeed::apply(),
              "Failed testing ScalarWave::ComputeLargestCharacteristicSpeed.");
