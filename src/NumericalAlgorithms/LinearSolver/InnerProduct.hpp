// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines an inner product for the linear solver

#pragma once

#include <array>

#include "DataStructures/Variables.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"

namespace LinearSolver {

/// \ingroup LinearSolverGroup
/// Implementations of LinearSolver::inner_product.
namespace InnerProductImpls {

/// The inner product between any types that have a `dot` product
template <typename Lhs, typename Rhs>
struct InnerProductImpl {
  static auto apply(const Lhs& lhs, const Rhs& rhs) {
    return dot(conj(lhs), rhs);
  }
};

/// The inner product between `Variables`
template <typename LhsTagsList, typename RhsTagsList>
struct InnerProductImpl<Variables<LhsTagsList>, Variables<RhsTagsList>> {
  using ValueType = typename Variables<LhsTagsList>::value_type;
  static ValueType apply(const Variables<LhsTagsList>& lhs,
                         const Variables<RhsTagsList>& rhs) {
    const auto size = lhs.size();
    ASSERT(size == rhs.size(),
           "The Variables must be of the same size to take an inner product");
    if constexpr (std::is_same_v<ValueType, std::complex<double>>) {
      return zdotc_(size, lhs.data(), 1, rhs.data(), 1);
    } else {
      return ddot_(size, lhs.data(), 1, rhs.data(), 1);
    }
  }
};

}  // namespace InnerProductImpls

/*!
 * \ingroup LinearSolverGroup
 * \brief The local part of the Euclidean inner product on the vector space
 * w.r.t. which the addition and scalar multiplication of both `Lhs` and `Rhs`
 * is defined.
 *
 * \details The linear solver works under the following assumptions:
 * - The data represented by \p lhs and \p rhs can each be interpreted as the
 * local chunk of a vector of the same vector space \f$V\f$. _Local_ means there
 * are vectors \f$q, p\in V\f$ such that \p lhs and \p rhs represent the
 * components of these vectors w.r.t. a subset \f$B_i\f$ of a basis
 * \f$B\subset V\f$.
 * - The `*` and `+` operators of `Lhs` and `Rhs` implement the scalar
 * multiplication and addition in the vector space _locally_, i.e. restricted to
 * \f$B_i\f$ in the above sense.
 * - The inner product is the local part \f$\langle p,q\rangle|_{B_i}\f$ of the
 * standard Euclidean dot product in the vector space so that globally it is
 * \f$\langle p,q\rangle=\sum_{i}\langle p,q\rangle|_{B_i}\f$ for
 * \f$B=\mathop{\dot{\bigcup}}_i B_i\f$.
 *
 * In practice this means that the full vectors \f$p\f$ and \f$q\f$ can be
 * distributed on many elements, where each only holds local chunks \p lhs and
 * \p rhs of the components. Scalar multiplication and addition can be performed
 * locally as expected, but computing the full inner product requires a global
 * reduction over all elements that sums their local `inner_product`s.
 */
template <typename Lhs, typename Rhs>
SPECTRE_ALWAYS_INLINE auto inner_product(const Lhs& lhs, const Rhs& rhs) {
  return InnerProductImpls::InnerProductImpl<Lhs, Rhs>::apply(lhs, rhs);
}

/*!
 * \ingroup LinearSolverGroup
 * \brief The local part of the Euclidean inner product of a vector with itself.
 *
 * \see LinearSolver::inner_product for details on the inner product. This
 * function just returns the inner product of a vector with itself, and ensures
 * that the result is real.
 */
template <typename T>
SPECTRE_ALWAYS_INLINE double magnitude_square(const T& vector) {
  auto result =
      InnerProductImpls::InnerProductImpl<T, T>::apply(vector, vector);
  if constexpr (std::is_same_v<std::decay_t<decltype(result)>,
                               std::complex<double>>) {
    ASSERT(equal_within_roundoff(imag(result), 0.0),
           "The magnitude squared is not real. The imaginary part is: "
               << imag(result));
    return real(result);
  } else {
    return result;
  }
}

}  // namespace LinearSolver
