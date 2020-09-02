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
#include "Evolution/Systems/ScalarWave/Equations.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
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
void check_normal_dot_fluxes(const size_t npts, const double t) {
  const ScalarWave::Solutions::PlaneWave<Dim> solution(
      make_array<Dim>(0.1), make_array<Dim>(0.0),
      std::make_unique<MathFunctions::Gaussian<1, Frame::Inertial>>(1.0, 1.0,
                                                                    0.0));

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
      make_not_null(&normal_dot_flux_psi), pi);

  const DataVector expected_normal_dot_fluxes{pow<Dim>(npts), 0.0};
  CHECK(get(normal_dot_flux_psi) == expected_normal_dot_fluxes);
  CHECK(get(normal_dot_flux_pi) == expected_normal_dot_fluxes);

  for (size_t d = 0; d < Dim; ++d) {
    CHECK(normal_dot_flux_phi.get(d) == expected_normal_dot_fluxes);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.NormalDotFluxes",
                  "[Unit][Evolution]") {
  constexpr double time = 0.7;
  INFO("Check NormalDotFluxes");
  check_normal_dot_fluxes<1>(3, time);
  check_normal_dot_fluxes<2>(3, time);
  check_normal_dot_fluxes<3>(3, time);
}

static_assert(1.0 == ScalarWave::ComputeLargestCharacteristicSpeed::apply(),
              "Failed testing ScalarWave::ComputeLargestCharacteristicSpeed.");
