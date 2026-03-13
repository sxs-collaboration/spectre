// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Jacobi.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

namespace {
template <size_t Dim>
void test_derivatives(const size_t num_points) {
  const Approx custom_approx =
      num_points > 9 ? Approx::custom().epsilon(1.0e-12).scale(1.0)
                      : Approx::custom().epsilon(1.0e-13).scale(1.0);
  // Testing differeniation for polynomials exactly representable, so
  // up to power = num_points
  constexpr Spectral::Basis basis = Dim == 1   ? Spectral::Basis::ZernikeB1
                                    : Dim == 2 ? Spectral::Basis::ZernikeB2
                                               : Spectral::Basis::ZernikeB3;
  CAPTURE(basis);
  CAPTURE(num_points);
  const Mesh<1> mesh{num_points, basis, Spectral::Quadrature::GaussRadauUpper};
  const auto coords = logical_coordinates(mesh);
  const DataVector r = 0.5 * (get<0>(coords) + 1.0);
  const auto f = [](const size_t power, const DataVector& x) {
    return integer_pow(x, static_cast<int>(power));
  };
  const auto d_f = [](const size_t power, const DataVector& x) {
    return static_cast<double>(power) *
           integer_pow(x, static_cast<int>(power) - 1);
  };
  auto& diff_matrix_even =
      Spectral::differentiation_matrix<basis,
                                       Spectral::Quadrature::GaussRadauUpper>(
          num_points, Spectral::Parity::Even);
  auto& diff_matrix_odd =
      Spectral::differentiation_matrix<basis,
                                       Spectral::Quadrature::GaussRadauUpper>(
          num_points, Spectral::Parity::Odd);
  for (size_t power = 0; power <= num_points; ++power) {
    CAPTURE(power);

    DataVector my_function(num_points);
    my_function = f(power, r);

    DataVector expected_derivative(num_points);
    expected_derivative =
        power > 0 ? d_f(power, r) : DataVector(num_points, 0.0);

    DataVector evaluated_derivative(num_points);
    // manually inserting 2 from above affine map (handled automatically
    // in partial_derivatives())
    evaluated_derivative =
        2.0 * (power % 2 == 0 ? apply_matrix(diff_matrix_even, my_function)
                              : apply_matrix(diff_matrix_odd, my_function));
    CHECK_ITERABLE_CUSTOM_APPROX(evaluated_derivative, expected_derivative,
                                 custom_approx);
  }
}

template <size_t Dim>
void test_weight_at_upper() {
  // used in dg::lift_flux()
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  const Spectral::Quadrature quadrature{Spectral::Quadrature::GaussRadauUpper};
  constexpr Spectral::Basis basis = Dim == 1   ? Spectral::Basis::ZernikeB1
                                    : Dim == 2 ? Spectral::Basis::ZernikeB2
                                               : Spectral::Basis::ZernikeB3;
  CAPTURE(basis);
  auto weight_at_upper = [](const size_t num_points) {
    if constexpr (Dim == 1) {
      return 1. / static_cast<double>(num_points * (2 * num_points - 1));
    } else if constexpr (Dim == 2) {
      return 1. / static_cast<double>(2 * square(num_points));
    } else {
      return 1. / static_cast<double>(num_points * (2 * num_points + 1));
    }
  };

  for (size_t n = 2; n < Spectral::maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const DataVector& weights =
        Spectral::quadrature_weights<basis, quadrature>(n);
    CHECK(weights[n - 1] == approx(weight_at_upper(n)));
  }
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.ZernikeGaussRadauUpper.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  // Being unaware of tabulated points and weights for Zernike
  // basis functions specifically with Gauss-Radau quadrature, we
  // test whether the basis can correctly take the derivatives of exactly
  // representable function
  for (size_t i = 1;
       i <= Spectral::maximum_number_of_points<Spectral::Basis::ZernikeB2>;
       ++i) {
    test_derivatives<1>(i);
    test_derivatives<2>(i);
    test_derivatives<3>(i);
  }
  test_weight_at_upper<1>();
  test_weight_at_upper<2>();
  test_weight_at_upper<3>();
}
