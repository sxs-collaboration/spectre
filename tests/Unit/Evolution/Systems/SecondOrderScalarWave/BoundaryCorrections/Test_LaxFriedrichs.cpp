// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections/LaxFriedrichs.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
Mesh<Dim - 1> face_mesh() {
  if constexpr (Dim == 1) {
    return Mesh<0>{};
  } else {
    return Mesh<Dim - 1>{5, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto};
  }
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen) {
  CAPTURE(Dim);
  PUPable_reg(SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>);

  std::uniform_real_distribution<> dist(0.0, 2.0);
  const double tau = dist(*gen);
  const SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>
      correction{tau};

  const auto mesh = face_mesh<Dim>();

  // The correction is a handful of O(1) multiply-adds, so C++ and python
  // agree to a few ULP; 1.0e-14 is tight while leaving ~10x flake margin.
  constexpr double eps = 1.0e-14;

  TestHelpers::evolution::dg::test_boundary_correction_conservation<
      SecondOrderScalarWave::System<Dim>>(
      gen, correction, mesh, {}, {},
      TestHelpers::evolution::dg::ZeroOnSmoothSolution::Yes, eps);
  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_conservation<
      SecondOrderScalarWave::System<Dim>>(
      gen, correction, mesh, {}, {},
      TestHelpers::evolution::dg::ZeroOnSmoothSolution::Yes, eps);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      SecondOrderScalarWave::System<Dim>>(
      gen, "LaxFriedrichs", "dg_package_data", "dg_boundary_terms", correction,
      mesh, {}, {}, eps, std::make_tuple(tau));
  TestHelpers::evolution::dg::test_auxiliary_boundary_correction_with_python<
      SecondOrderScalarWave::System<Dim>>(
      gen, "LaxFriedrichs", "dg_auxiliary_package_data",
      "dg_auxiliary_boundary_terms", correction, mesh, {}, {}, eps,
      std::make_tuple(tau));

  // Factory creation round-trips the options into an equal object.
  const auto created = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>>(
      "LaxFriedrichs:\n"
      "  Tau: 1.5\n");
  const auto& downcast = dynamic_cast<
      const SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>&>(
      *created);
  CHECK(downcast ==
        SecondOrderScalarWave::BoundaryCorrections::LaxFriedrichs<Dim>{1.5});

  // Pin the return values of the two package-data functions: the physical pass
  // returns the unit wave speed, the auxiliary pass returns a signaling NaN.
  const size_t num_pts = 1;
  const Scalar<DataVector> psi{num_pts, 0.5};
  const Scalar<DataVector> pi{num_pts, 0.5};
  const tnsr::i<DataVector, Dim, Frame::Inertial> phi{num_pts, 0.5};
  tnsr::i<DataVector, Dim, Frame::Inertial> normal_covector{num_pts, 0.0};
  get<0>(normal_covector) = 1.0;

  Scalar<DataVector> packaged_pi{num_pts};
  Scalar<DataVector> packaged_normal_dot_phi{num_pts};
  const double max_char_speed = correction.dg_package_data(
      make_not_null(&packaged_pi), make_not_null(&packaged_normal_dot_phi), psi,
      pi, phi, normal_covector, std::nullopt, std::nullopt);
  CHECK(max_char_speed == 1.0);

  tnsr::i<DataVector, Dim, Frame::Inertial> psi_times_normal{num_pts};
  const double auxiliary_speed = correction.dg_auxiliary_package_data(
      make_not_null(&psi_times_normal), psi, pi, normal_covector, std::nullopt,
      std::nullopt);
  {
    const ScopedFpeState disable_fpes(false);
    CHECK(std::isnan(auxiliary_speed));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.SecondOrderScalarWave.BoundaryCorrections.LaxFriedrichs",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SecondOrderScalarWave/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen));
  test<2>(make_not_null(&gen));
  test<3>(make_not_null(&gen));
}
