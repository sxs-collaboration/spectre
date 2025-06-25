// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

// The Basis::Cartoon should never be called to generate collocation points or
// weights, so it errors on everything

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonAxialSymmetry.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::collocation_points<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
  CHECK_THROWS_WITH(
      (Spectral::quadrature_weights<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::differentiation_matrix<
          Spectral::Basis::Cartoon, Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.ModalToNodal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::modal_to_nodal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.NodalToModal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::nodal_to_modal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::collocation_points<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
  CHECK_THROWS_WITH(
      (Spectral::quadrature_weights<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonSphericalSymmetry.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::differentiation_matrix<
          Spectral::Basis::Cartoon, Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.ModalToNodal",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::modal_to_nodal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.NodalToModal",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::nodal_to_modal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}
