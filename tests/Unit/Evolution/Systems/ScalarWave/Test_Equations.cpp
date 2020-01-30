// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

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
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/DiscontinuousGalerkin/NumericalFluxes/TestHelpers.hpp"
#include "Helpers/Utilities/ProtocolTestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Protocols.hpp"
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
auto add_scalar_to_tensor_components(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& input,
    double constant) noexcept {
  const tnsr::i<DataVector, Dim, Frame::Inertial> copy_of_input =
      [&input, &constant ]() noexcept {
    auto local_copy_of_input = input;
    for (size_t i = 0; i < Dim; ++i) {
      local_copy_of_input.get(i) += constant;
    }
    return local_copy_of_input;
  }
  ();
  return copy_of_input;
}
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

  auto local_check_du_dt = [&npts, &time, &solution, &x](
                               const double gamma2,
                               const double constraint) {
    auto box = db::create<db::AddSimpleTags<
        ScalarWave::Tags::ConstraintGamma2, Tags::dt<ScalarWave::Pi>,
        Tags::dt<ScalarWave::Phi<Dim>>, Tags::dt<ScalarWave::Psi>,
        ScalarWave::Pi, ScalarWave::Phi<Dim>,
        Tags::deriv<ScalarWave::Pi, tmpl::size_t<Dim>, Frame::Inertial>,
        Tags::deriv<ScalarWave::Psi, tmpl::size_t<Dim>, Frame::Inertial>,
        Tags::deriv<ScalarWave::Phi<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>>(
        Scalar<DataVector>(pow<Dim>(npts), gamma2),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        tnsr::i<DataVector, Dim, Frame::Inertial>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(pow<Dim>(npts), 0.0),
        Scalar<DataVector>(-1.0 * solution.dpsi_dt(x, time).get()),
        add_scalar_to_tensor_components(solution.dpsi_dx(x, time),
                                        constraint),
        [&x, &time, &solution ]() noexcept {
          auto dpi_dx = solution.d2psi_dtdx(x, time);
          for (size_t i = 0; i < Dim; ++i) {
            dpi_dx.get(i) *= -1.0;
          }
          return dpi_dx;
        }(),
        solution.dpsi_dx(x, time), [&npts, &x, &time, &solution ]() noexcept {
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
    CHECK_ITERABLE_APPROX(
        db::get<Tags::dt<ScalarWave::Phi<Dim>>>(box),
        add_scalar_to_tensor_components(solution.d2psi_dtdx(x, time),
                                        -gamma2 * constraint));
    CHECK_ITERABLE_APPROX(db::get<Tags::dt<ScalarWave::Psi>>(box),
                          solution.dpsi_dt(x, time));
  };

  // Test with constraint damping parameter set to zero
  local_check_du_dt(0.0, 0.0);
  local_check_du_dt(0.0, 100.0);
  local_check_du_dt(0.0, -998.0);

  // Test with constraint satisfied but nonzero constraint damping parameter
  local_check_du_dt(10.0, 0.0);
  local_check_du_dt(-4.3, 0.0);

  // Test with one-index constraint NOT satisfied and nonzero constraint
  // damping parameter
  local_check_du_dt(10.0, 10.9);
  local_check_du_dt(1.2, -77.0);
  local_check_du_dt(-10.9, 43.0);
  local_check_du_dt(-90.0, -56.0);
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
template <size_t Dim>
void check_upwind_flux(const size_t npts, const double t) {
  static_assert(test_protocol_conformance<ScalarWave::UpwindFlux<Dim>,
                                          dg::protocols::NumericalFlux>,
                "Failed testing protocol conformance");

  const DataVector used_for_size{pow<Dim>(npts),
                                 std::numeric_limits<double>::signaling_NaN()};
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

  ScalarWave::UpwindFlux<Dim> flux_computer{};

  auto packaged_data_int = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, solution.dpsi_dt(x, t + 1.0),
      solution.dpsi_dx(x, t + 2.0), solution.psi(x, t + 4.0),
      solution.psi(x, t + 4.0), Scalar<DataVector>(pow<Dim>(npts), 0.0),
      unit_normal);
  auto packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, solution.dpsi_dt(x, 2.0 * t + 10.0),
      solution.dpsi_dx(x, 2.0 * t + 9.0), solution.psi(x, 2.0 * t + 7.0),
      solution.psi(x, t + 4.0), Scalar<DataVector>(pow<Dim>(npts), 0.0),
      unit_normal);

  Scalar<DataVector> normal_dot_numerical_flux_pi(pow<Dim>(npts), 0.0);
  Scalar<DataVector> normal_dot_numerical_flux_psi(pow<Dim>(npts), 0.0);
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_dot_numerical_flux_phi(
      pow<Dim>(npts), 0.0);
  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      flux_computer, packaged_data_int, packaged_data_ext,
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

namespace {
template <size_t Dim>
void penalty_flux(
    const gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_pi,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        normal_dot_numerical_flux_phi,
    const gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux_psi,
    const Scalar<DataVector>& n_dot_flux_pi_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& n_dot_flux_phi_int,
    const Scalar<DataVector>& v_plus_int, const Scalar<DataVector>& v_minus_int,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_int,
    const Scalar<DataVector>& minus_n_dot_flux_pi_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& minus_n_dot_flux_phi_ext,
    const Scalar<DataVector>& v_plus_ext, const Scalar<DataVector>& v_minus_ext,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_ext) noexcept {
  const size_t num_pts = v_plus_int.begin()->size();
  const DataVector used_for_size{num_pts,
                                 std::numeric_limits<double>::signaling_NaN()};

  ScalarWave::PenaltyFlux<Dim> flux_computer{};

  auto packaged_data_int = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, n_dot_flux_pi_int, n_dot_flux_phi_int,
      v_plus_int, v_minus_int, unit_normal_int);
  auto packaged_data_ext = TestHelpers::NumericalFluxes::get_packaged_data(
      flux_computer, used_for_size, minus_n_dot_flux_pi_ext,
      minus_n_dot_flux_phi_ext, v_plus_ext, v_minus_ext, unit_normal_ext);

  dg::NumericalFluxes::normal_dot_numerical_fluxes(
      flux_computer, packaged_data_int, packaged_data_ext,
      normal_dot_numerical_flux_pi, normal_dot_numerical_flux_phi,
      normal_dot_numerical_flux_psi);
}

template <size_t Dim>
void check_penalty_flux(const size_t num_pts_per_dim) noexcept {
  static_assert(test_protocol_conformance<ScalarWave::PenaltyFlux<Dim>,
                                          dg::protocols::NumericalFlux>,
                "Failed testing protocol conformance");

  pypp::check_with_random_values<10>(
      &penalty_flux<Dim>, "PenaltyFlux",
      {"pi_penalty_flux", "phi_penalty_flux", "psi_penalty_flux"},
      {{{-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0},
        {-1.0, 1.0}}},
      DataVector{pow<Dim>(num_pts_per_dim)});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.PenaltyFlux",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave"};

  constexpr size_t num_pts_per_dim = 5;
  check_penalty_flux<1>(num_pts_per_dim);
  check_penalty_flux<2>(num_pts_per_dim);
  check_penalty_flux<3>(num_pts_per_dim);
}

static_assert(1.0 == ScalarWave::ComputeLargestCharacteristicSpeed::apply(),
              "Failed testing ScalarWave::ComputeLargestCharacteristicSpeed.");
