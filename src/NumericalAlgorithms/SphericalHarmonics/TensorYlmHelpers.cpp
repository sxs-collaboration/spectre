// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmHelpers.hpp"

#include <array>
#include <complex>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Array.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace ylm::TensorYlm::helpers {

template <size_t Rank>
std::array<BasisVector, Rank> to_cart_basis_vector(
    const cpp20::array<size_t, Rank>& indices) {
  static_assert(Rank <= 3, "Implemented only for rank up to 3");
  std::array<BasisVector, Rank> result{};
  for (size_t i = 0; i < Rank; ++i) {
    switch (indices[i]) {
      case 0:
        gsl::at(result, i) = BasisVector::x;
        break;
      case 1:
        gsl::at(result, i) = BasisVector::y;
        break;
      case 2:
        gsl::at(result, i) = BasisVector::z;
        break;
      default:
        ASSERT(false, "Cannot get here");
    }
  }
  return result;
}

int bv_to_m(const BasisVector basis_vector, const int i) {
  int result = std::numeric_limits<int>::min();
  switch (basis_vector) {
    case BasisVector::z:
      result = 0;
      break;
    case BasisVector::y:
    case BasisVector::x:
      result = i;
      break;
    default:
      ASSERT(false, "Unknown basisvector");
  }
  return result;
}

std::complex<double> bv_to_k(const BasisVector basis_vector, const int i) {
  std::complex<double> result{std::numeric_limits<double>::signaling_NaN(),
                              std::numeric_limits<double>::signaling_NaN()};
  switch (basis_vector) {
    case BasisVector::z:
      result = {1.0 / sqrt(2.0), 0.0};
      break;
    case BasisVector::y:
      result = {0, 1};
      break;
    case BasisVector::x:
      result = {double(-i), 0.0};
      break;
    default:
      ASSERT(false, "Unknown basisvector");
  }
  return result;
}

template <size_t Rank>
std::array<BasisVector, Rank> to_sphere_basis_vector(
    const cpp20::array<size_t, Rank>& indices) {
  static_assert(Rank <= 3, "Implemented only for rank up to 3");
  std::array<BasisVector, Rank> result{};
  for (size_t i = 0; i < Rank; ++i) {
    switch (indices[i]) {
      case 0:
        gsl::at(result, i) = BasisVector::l;
        break;
      case 1:
        gsl::at(result, i) = BasisVector::m;
        break;
      case 2:
        gsl::at(result, i) = BasisVector::mbar;
        break;
      default:
        ASSERT(false, "Cannot get here");
    }
  }
  return result;
}

int bv_to_s(const BasisVector basis_vector) {
  int result = std::numeric_limits<int>::min();
  switch (basis_vector) {
    case BasisVector::l:
      result = 0;
      break;
    case BasisVector::m:
      result = -1;
      break;
    case BasisVector::mbar:
      result = 1;
      break;
    default:
      ASSERT(false, "Unknown basisvector " << static_cast<int>(basis_vector));
  }
  return result;
}

template <typename Symm>
double get_symm_factor(const size_t src_multiplicity, const size_t lbar) {
  static_assert(std::is_same_v<Symmetry<3, 2, 1>, Symm> or
                    std::is_same_v<Symmetry<2, 1, 1>, Symm> or
                    std::is_same_v<Symmetry<2, 1>, Symm> or
                    std::is_same_v<Symmetry<1, 1>, Symm>,
                "Unimplemented symmetry");
  if constexpr (std::is_same_v<Symmetry<3, 2, 1>, Symm> or
                std::is_same_v<Symmetry<2, 1>, Symm>) {
    // Not symmetric on the 2 indices.
    (void)src_multiplicity;
    (void)lbar;
  } else {
    // symmetric on the 2 indices.
    //
    // Note that src_multiplicity is 1 if the indices are the same,
    // 2 if they are different.  For more complicated symmetries, if
    // we ever wanted to implement them, this formula would be different.
    return (lbar % 2 == 0 ? static_cast<double>(src_multiplicity) : 0.0);
  }
  return 1.0;
}

// Explicit instantiations
template std::array<BasisVector, 1> to_cart_basis_vector(
    const cpp20::array<size_t, 1>& indices);
template std::array<BasisVector, 2> to_cart_basis_vector(
    const cpp20::array<size_t, 2>& indices);
template std::array<BasisVector, 3> to_cart_basis_vector(
    const cpp20::array<size_t, 3>& indices);
template std::array<BasisVector, 1> to_sphere_basis_vector(
    const cpp20::array<size_t, 1>& indices);
template std::array<BasisVector, 2> to_sphere_basis_vector(
    const cpp20::array<size_t, 2>& indices);
template std::array<BasisVector, 3> to_sphere_basis_vector(
    const cpp20::array<size_t, 3>& indices);
template double get_symm_factor<Symmetry<3, 2, 1>>(size_t src_multiplicity,
                                                   size_t lbar);
template double get_symm_factor<Symmetry<2, 1, 1>>(size_t src_multiplicity,
                                                   size_t lbar);
template double get_symm_factor<Symmetry<1, 1>>(size_t src_multiplicity,
                                                size_t lbar);
template double get_symm_factor<Symmetry<2, 1>>(size_t src_multiplicity,
                                                size_t lbar);

}  // namespace ylm::TensorYlm::helpers
