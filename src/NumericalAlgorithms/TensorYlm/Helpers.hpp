// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/TensorYlm/TensorYlm.hpp"

#include <array>
#include <complex>
#include <cstdint>

#include "Utilities/Array.hpp"

namespace ylm::TensorYlm::helpers {

/// Defines the three Cartesian and three spherical basis vectors.
enum class BasisVector : uint8_t { x, y, z, l, m, mbar };

/// Returns a Cartesian BasisVector for every index.
template <size_t Rank>
std::array<BasisVector, Rank> to_cart_basis_vector(
    const cpp20::array<size_t, Rank>& indices);

/// Returns the m value associated with a Cartesian basis vector.
int bv_to_m(BasisVector basis_vector, int i);

/// Returns the prefactor k associated with a Cartesian basis vector.
std::complex<double> bv_to_k(BasisVector basis_vector, int i);

/// Returns a spherical BasisVector for every index.
template <size_t Rank>
std::array<BasisVector, Rank> to_sphere_basis_vector(
    const cpp20::array<size_t, Rank>& indices);

/// Returns minus the spinweight of a tetrad basis vector
int bv_to_s(BasisVector basis_vector);

/// Computes the symmetry factor S that appears in various equations.
template <typename Symm>
double get_symm_factor(size_t src_multiplicity, size_t lbar);

/// Computes the (-1)^m factor that is used depending on whether the
/// coefficient normalization is Spherepack.
template <typename T>
constexpr T sign_m(const int m,
                   const CoefficientNormalization coefficient_normalization) {
  return coefficient_normalization == CoefficientNormalization::Spherepack and
                 m % 2 != 0
             ? static_cast<T>(-1)
             : static_cast<T>(1);
}

}  // namespace ylm::TensorYlm::helpers
