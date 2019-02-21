// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <complex>
#include <cstddef>
#include <sharp_cxx.h>

#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/Spectral/ComplexDataView.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshTags.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "tests/Utilities/MakeWithRandomValues.hpp"

/// \cond
class ComplexDataVector;
/// \endcond

namespace Spectral {
namespace Swsh {
namespace TestHelpers {

// returns the factorial of the argument as a double so that an approximate
// value can be given for larger input quantities. Note that the spin-weighted
// harmonic function requires a factorial of l + m, so harmonics above l~12
// would be untestable if the factorial only returned size_t's.
double factorial(size_t arg) noexcept;

// Note that the methods for computing the spin-weighted spherical harmonics and
// their derivatives below are both 1) poorly optimized (they use many
// computations per grid point evaluated) and 2) not terribly accurate (the
// analytic expressions require evaluation of ratios of factorials, losing
// numerical precision rapidly). However, they are comparatively easy to
// manually check for correctness, which is critical to offer a reliable measure
// of the spin-weighted transforms.

// Analytic form for the spin-weighted spherical harmonic function, for testing
// purposes. The formula is from [Goldberg
// et. al.](https://aip.scitation.org/doi/10.1063/1.1705135)
std::complex<double> spin_weighted_spherical_harmonic(int s, int l, int m,
                                                      double theta,
                                                      double phi) noexcept;

// Returns the value of the spin-weighted derivative `DerivativeKind` of the
// spherical harmonic basis function \f${}_s Y_{l m}\f$ at angular location
// (`theta`, `phi`) using the recurrence identities for spin-weighted
// derivatives.
template <typename DerivativeKind>
std::complex<double> derivative_of_spin_weighted_spherical_harmonic(
    int s, int l, int m, double theta, double phi) noexcept;

// Performs the generation of spin-weighted coefficients into a supplied
// ComplexModalVector `to_fill`, passed by pointer. The resulting random
// coefficients must then be adjusted in two ways:
// - The modes l < |s| are always zero, so must have zero coefficient to be
//   compatible with tests
// - Modes with m=0 obey reality conditions, so must also be adjusted
// The libsharp coefficient representation is not currently presented as an
// interface in SpECTRE. However, to help maintain the Swsh library, further
// notes on the coefficient representation can be found in the comments for the
// `Coefficients` class in `SwshCoefficients.hpp`.
template <int Spin, typename Distribution, typename Generator>
void generate_swsh_modes(const gsl::not_null<ComplexModalVector*> to_fill,
                         const gsl::not_null<Generator*> generator,
                         const gsl::not_null<Distribution*> distribution,
                         const size_t number_of_radial_points,
                         const size_t l_max) noexcept {
  fill_with_random_values(to_fill, generator, distribution);

  auto spherical_harmonic_lm =
      detail::precomputed_coefficients(l_max).get_sharp_alm_info();
  for (size_t i = 0; i < number_of_radial_points; i++) {
    // adjust the m = 0 modes for the reality conditions in libsharp
    for (size_t l = 0; l < l_max + 1; l++) {
      (*to_fill)[l + 2 * i * number_of_swsh_coefficients(l_max)] =
          real((*to_fill)[l + i * number_of_swsh_coefficients(l_max)]);
      (*to_fill)[l + (2 * i + 1) * number_of_swsh_coefficients(l_max)] = imag(
          (*to_fill)[l + (2 * i + 1) * number_of_swsh_coefficients(l_max)]);
    }
    for (size_t m = 0; m < static_cast<size_t>(spherical_harmonic_lm->nm);
         ++m) {
      for (size_t l = m; l <= l_max; ++l) {
        // clang-tidy do not use pointer arithmetic
        // pointer arithmetic here is unavoidable as we are interacting with
        // the libsharp structures
        const auto m_start =
            static_cast<size_t>(spherical_harmonic_lm->mvstart[m]);  // NOLINT
        const auto l_stride =
            static_cast<size_t>(spherical_harmonic_lm->stride);
        // modes for l < |s| are zero
        if (static_cast<int>(l) < abs(Spin)) {
          (*to_fill)[m_start + l * l_stride +
                     2 * i * number_of_swsh_coefficients(l_max)] = 0.0;
          (*to_fill)[m_start + l * l_stride +
                     (2 * i + 1) * number_of_swsh_coefficients(l_max)] = 0.0;
        }
      }
    }
  }
}

// Computes a set of collocation points from a set of spin-weighted spherical
// harmonic coefficients. This function takes care of the nastiness associated
// with appropriately iterating over the sharp representation of coefficients
// and storing the result computed from the input `basis_function` evaluated
// with arguments (size_t s, size_t l, size_t m, double theta, double phi),
// which is compatible with the above `spin_weighted_spherical_harmonic`
// and `derivative_of_spin_weighted_spherical_harmonic` functions.
// This function does not have easy test verification, but acts as an
// independent computation of the spherical harmonic collocation values, so the
// agreement (from `Test_SwshTransformJob.cpp`) with the libsharp transform
// lends credibility to both methods.
template <int Spin, ComplexRepresentation Representation,
          typename BasisFunction>
void swsh_collocation_from_coefficients_and_basis_func(
    const gsl::not_null<ComplexDataVector*> collocation_data,
    const gsl::not_null<ComplexModalVector*> coefficient_data,
    const size_t l_max, const size_t number_of_radial_points,
    const BasisFunction basis_function) noexcept {
  auto& spherical_harmonic_collocation =
      precomputed_collocation<Representation>(l_max);
  auto spherical_harmonic_lm =
      detail::precomputed_coefficients(l_max).get_sharp_alm_info();

  for (size_t i = 0; i < number_of_radial_points; ++i) {
    for (auto j : spherical_harmonic_collocation) {
      for (size_t m = 0; m < static_cast<size_t>(spherical_harmonic_lm->nm);
           ++m) {
        for (size_t l = m; l <= l_max; ++l) {
          // clang-tidy do not use pointer arithmetic
          // pointer arithmetic here is unavoidable as we are interacting with
          // the libsharp structures
          const auto m_start =
              static_cast<size_t>(spherical_harmonic_lm->mvstart[m]);  // NOLINT
          const auto l_stride =
              static_cast<size_t>(spherical_harmonic_lm->stride);
          // see the documentation associated with Swsh::TransformJob in
          // SwshTransformJob.hpp for a detailed explanation of the libsharp
          // collocation representation, and why there must be four steps with
          // sign transformations.
          (*collocation_data)[j.offset +
                              i * spherical_harmonic_collocation.size()] +=
              sharp_swsh_sign(Spin, m, true) *
              (*coefficient_data)[m_start + l * l_stride +
                                  2 * i * number_of_swsh_coefficients(l_max)] *
              basis_function(Spin, l, m, j.theta, j.phi);

          if (m != 0) {
            (*collocation_data)[j.offset +
                                i * spherical_harmonic_collocation.size()] +=
                sharp_swsh_sign(Spin, -m, true) *
                conj((*coefficient_data)
                         [m_start + l * l_stride +
                          2 * i * number_of_swsh_coefficients(l_max)]) *
                basis_function(Spin, l, -m, j.theta, j.phi);
          }
          (*collocation_data)[j.offset +
                              i * spherical_harmonic_collocation.size()] +=
              sharp_swsh_sign(Spin, m, false) * std::complex<double>(0.0, 1.0) *
              (*coefficient_data)[m_start + l * l_stride +
                                  (2 * i + 1) *
                                      number_of_swsh_coefficients(l_max)] *
              basis_function(Spin, l, m, j.theta, j.phi);
          if (m != 0) {
            (*collocation_data)[j.offset +
                                i * spherical_harmonic_collocation.size()] +=
                sharp_swsh_sign(Spin, -m, false) *
                std::complex<double>(0.0, 1.0) *
                conj((*coefficient_data)
                         [m_start + l * l_stride +
                          (2 * i + 1) * number_of_swsh_coefficients(l_max)]) *
                basis_function(Spin, l, -m, j.theta, j.phi);
          }
        }
      }
    }
  }
}
}  // namespace TestHelpers
}  // namespace Swsh
}  // namespace Spectral
