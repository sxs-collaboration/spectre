// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/ArrayCollection/DgElementArrayMemberBase.hpp"
#include "Parallel/ArrayCollection/IsDgElementArrayMember.hpp"

namespace Parallel {
SPECTRE_TEST_CASE("Unit.Parallel.ArrayCollection.IsDgElementArrayMember",
                  "[Unit][Parallel]") {
  CHECK(is_dg_element_array_member_v<DgElementArrayMemberBase<3>>);
  CHECK(
      is_dg_element_array_member_v<DgElementArrayMember<3, void, void, void>>);
  CHECK_FALSE(is_dg_element_array_member_v<int>);
}
}  // namespace Parallel
