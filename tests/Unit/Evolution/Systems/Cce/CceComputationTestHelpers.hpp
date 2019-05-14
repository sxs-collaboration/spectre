// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestingFramework.hpp"

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

// A utility for copying a set of tags from one DataBox to another. Useful in
// the Cce tests, where often an expected input needs to be copied to the box
// which is to be tested.
template <typename... Tags>
struct CopyDataBoxTags {
  template <typename FromDataBox, typename ToDataBox>
  static void apply(const gsl::not_null<ToDataBox*> to_data_box,
                    const FromDataBox& from_data_box) noexcept {
    db::mutate<Tags...>(
        to_data_box,
        [](const gsl::not_null<db::item_type<Tags>*>... to_value,
           const typename Tags::type&... from_value) {
          auto assign = [](auto to, auto from) {
            *to = from;
            return 0;
          };
          expand_pack(assign(to_value, from_value)...);
        },
        db::get<Tags>(from_data_box)...);
  }
};
}  // namespace TestHelpers
}  // namespace Cce
