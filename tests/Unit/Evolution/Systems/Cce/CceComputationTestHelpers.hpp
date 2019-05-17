// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class ComplexDataVector;
class ComplexModalVector;
/// \endcond

namespace Cce {
namespace TestHelpers {

// For representing a primitive series of powers in inverse r for diagnostic
// computations
template <typename Tag>
struct RadialPolyCoefficientsFor : db::SimpleTag, db::PrefixTag {
  using type = Scalar<ComplexModalVector>;
  using tag = Tag;
};

// For representing the angular function in a separable quantity.
template <typename Tag>
struct AngularCollocationsFor : db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

// shortcut method for evaluating a frequently used radial quantity in CCE
void volume_one_minus_y(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> one_minus_y,
    size_t l_max) noexcept;

// explicit power method avoids behavior of Blaze to occasionally FPE on
// powers of complex that operate fine when repeated multiplication is used
// instead.
ComplexDataVector power(const ComplexDataVector& value,
                        size_t exponent) noexcept;

}  // namespace TestHelpers
}  // namespace Cce
