// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace Spectral {
namespace {
template <Basis basis, Quadrature quadrature>
void test() {
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const DataVector xi = collocation_points<basis, quadrature>(n);
    const Matrix m = nodal_to_modal_matrix<basis, quadrature>(n);
    for (size_t k = 0; k < n; ++k) {
      CAPTURE(k);
      const auto f = compute_basis_function_value<basis>(k, xi);
      const auto f_k = apply_matrix(m, f);
      DataVector f_k_expected{n, 0.0};
      f_k_expected[k] = 1.0;
      CHECK_ITERABLE_APPROX(f_k, f_k_expected);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.NodalToModalMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test<Basis::Fourier, Quadrature::Equiangular>();
}
}  // namespace Spectral
