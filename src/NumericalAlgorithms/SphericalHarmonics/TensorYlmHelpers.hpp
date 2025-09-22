// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"

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

/// Computes the symmetry factor S that appears in various equations.
template <typename Symm>
double get_symm_factor(size_t src_multiplicity, size_t lbar);

}  // namespace ylm::TensorYlm::helpers
