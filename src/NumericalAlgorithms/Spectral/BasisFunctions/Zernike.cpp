// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/Blaze/IntegerPow.hpp"
#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Math.hpp"

namespace Spectral {
namespace {
template <size_t Dim, typename T>
T Q(const size_t n, const size_t m, const T& r) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "m, " << m << ", must be at most n, " << n);
  ASSERT((n + m) % 2 == 0,
         "n, " << n << ", plus m, " << m << ", must be even.");
  const auto mm = static_cast<double>(m);

  // Implemented from Matsushima1995
  // Matsushima and Marcus paper notation varries from usual standards
  // Note: their \alpha and \beta are NOT Jacobi polynomial \alpha and \beta
  // For 1D, their \alpha = 1, \beta = 0
  // For 2D, their \alpha = 1, \beta = 1
  // For 3D, their \alpha = 1, \beta = 2
  // This means their \gamma := 2\alpha + \beta is either 3 for polar or
  // 4 for spherical, which in SpEC's mBeta = \gamma - 2
  constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
  // Qnm is Q_n^m, p2/m2 being plus2 / minus2
  // using betaM to indicate Marcus Beta
  T Qmm = integer_pow(r, m);
  if (n == m) {
    return Qmm;
  }
  T Qmp2m =
      0.5 * Qmm * ((2. * mm + betaM + 3.) * square(r) - (2. * mm + betaM + 1.));
  if (n == (m + 2)) {
    return Qmp2m;
  }
  T Qnm = Qmp2m;
  T Qnm2m = Qmm;
  for (size_t n_loop = m + 2; n_loop < n; n_loop += 2) {
    const auto nn = static_cast<double>(n_loop);
    const T Qnp2m =
        ((2. * nn + betaM + 1.) *
             ((2. * nn + betaM - 1.) * (2. * nn + betaM + 3.) * square(r) -
              2. * nn * (nn + betaM + 1.) - 2. * mm * (mm + betaM - 1.) -
              (betaM - 1.) * (betaM + 1.)) *
             Qnm -
         (nn - mm) * (nn + mm + betaM - 1.) * (2. * nn + betaM + 3.) * Qnm2m) /
        ((nn - mm + 2.) * (nn + mm + betaM + 1.) * (2. * nn + betaM - 1.));
    Qnm2m = Qnm;
    Qnm = Qnp2m;
  }
  return Qnm;
}

template <size_t Dim>
double I(const size_t n, const size_t m) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  ASSERT(n >= m, "n = " << n << "; m = " << m);
  ASSERT((n + m) % 2 == 0,
         "n and m must have same parity, got n = " << n << " m = " << m);
  constexpr double betaM = Dim == 1 ? 0.0 : (Dim == 2 ? 1.0 : 2.0);
  double result = 0.5 / (static_cast<double>(m) + 0.5 + 0.5 * betaM);
  for (size_t i = m + 2; i <= n; i += 2) {
    const auto ii = static_cast<double>(i);
    result *= (2. * ii + betaM - 3.) / (2. * ii + betaM + 1.);
  }
  return result;
}
}  // namespace

template <size_t Dim>
template <typename T>
T Zernike<Dim>::basis_function_value(const size_t n, const size_t m,
                                     const T& xi) {
  static_assert(Dim == 1 or Dim == 2 or Dim == 3);
  // the sqrt(I) normalization is for normalizing "with respect to the weight
  // w(r)," which is the integrating weight in the orthogonality condition
  // Note that xi is logical coordinate, so in [-1, 1]. Need to shift back to
  // true [0, 1] range
  return Q<Dim>(n, m, static_cast<T>(0.5 * (xi + 1.0))) / sqrt(I<Dim>(n, m));
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_TYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_BASIS_FUNCTION_VALUE(r, data)                       \
  template GET_TYPE(data) Zernike<GET_DIM(data)>::basis_function_value( \
      const size_t n, const size_t m, const GET_TYPE(data) & xi);

GENERATE_INSTANTIATIONS(INSTANTIATE_BASIS_FUNCTION_VALUE, (1, 2, 3),
                        (double, DataVector))

}  // namespace Spectral
