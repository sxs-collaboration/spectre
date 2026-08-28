// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/UpdateAuxiliaryVariables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void check_update_aux(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const Mesh<Dim> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const size_t num_pts = mesh.number_of_grid_points();

  const auto psi = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&distribution), DataVector(num_pts));

  const std::array<double, 3> scales{{0.5, 1.5, 2.5}};
  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{num_pts, 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 1.0 / gsl::at(scales, i);
  }

  const auto expected_phi = partial_derivative(psi, mesh, inv_jac);

  auto box = db::create<db::AddSimpleTags<
      SecondOrderScalarWave::Tags::Psi, SecondOrderScalarWave::Tags::Phi<Dim>,
      domain::Tags::Mesh<Dim>,
      domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                    Frame::Inertial>>>(
      psi, tnsr::i<DataVector, Dim, Frame::Inertial>{num_pts, 0.0}, mesh,
      inv_jac);
  db::mutate_apply<SecondOrderScalarWave::UpdateAuxiliaryVariables<Dim>>(
      make_not_null(&box));

  CHECK_ITERABLE_APPROX((db::get<SecondOrderScalarWave::Tags::Phi<Dim>>(box)),
                        expected_phi);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.SecondOrderScalarWave.UpdateAuxiliaryVariables",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(generator);
  check_update_aux<1>(make_not_null(&generator));
  check_update_aux<2>(make_not_null(&generator));
  check_update_aux<3>(make_not_null(&generator));
}
