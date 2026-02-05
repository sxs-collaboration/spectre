// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Helpers/DataStructures/ApplyMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
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
    const Matrix m = modal_to_nodal_matrix<basis, quadrature>(n);
    for (size_t k = 0; k < n; ++k) {
      CAPTURE(k);
      const auto f_expected = compute_basis_function_value<basis>(k, xi);
      DataVector f_k{n, 0.0};
      f_k[k] = 1.0;
      const auto f = apply_matrix(m, f_k);
      CHECK_ITERABLE_APPROX(f, f_expected);
    }
  }
}

template <Basis basis, Quadrature quadrature>
void test_two_indexed() {
  static_assert(basis == Basis::ZernikeB1 or basis == Basis::ZernikeB2 or
                basis == Basis::ZernikeB3);
  CAPTURE(basis);
  CAPTURE(quadrature);
  for (size_t n = minimum_number_of_points<basis, quadrature>;
       n <= maximum_number_of_points<basis>; ++n) {
    CAPTURE(n);
    const DataVector xi = collocation_points<basis, quadrature>(n);
    for (size_t N = 0; N < 2 * n - 1; ++N) {
      CAPTURE(N);
      for (size_t m = 0; m <= N; ++m) {
        CAPTURE(m);
        const Matrix modal_to_nodal =
            modal_to_nodal_matrix<basis, quadrature>(n, m, N);
        // The spectal space is only odd or even modes, based on m parity
        // Note the integer division
        const size_t spectral_size = (N - m) / 2 + 1;
        for (size_t k = m; k <= N; k += 2) {
          CAPTURE(k);
          const auto f_expected = compute_basis_function_value<basis>(k, m, xi);
          DataVector f_k{spectral_size, 0.0};
          // Index in this compressed space (of specific parity) must be mapped
          const size_t index = (k - m) / 2;
          f_k[index] = 1.0;
          const auto f = apply_matrix(modal_to_nodal, f_k);
          CHECK_ITERABLE_APPROX(f, f_expected);
        }
      }
    }
  }
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.Numerical.Spectral.ModalToNodalMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  test<Basis::Legendre, Quadrature::Gauss>();
  test<Basis::Legendre, Quadrature::GaussLobatto>();
  test<Basis::Chebyshev, Quadrature::Gauss>();
  test<Basis::Chebyshev, Quadrature::GaussLobatto>();
  test<Basis::Fourier, Quadrature::Equiangular>();
  test_two_indexed<Basis::ZernikeB1, Quadrature::GaussRadauUpper>();
  test_two_indexed<Basis::ZernikeB2, Quadrature::GaussRadauUpper>();
  test_two_indexed<Basis::ZernikeB3, Quadrature::GaussRadauUpper>();
}
}  // namespace Spectral
