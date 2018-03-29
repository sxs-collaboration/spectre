// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/StripeIterator.hpp"

namespace {
void check_stripe_iterator_helper(StripeIterator s) {
  CHECK((s.stride() == 3 and s.offset() == 0 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 1 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 2 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 12 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 13 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 14 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 24 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 25 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 26 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 36 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 37 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 38 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 48 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 49 and s and ++s));
  CHECK((s.stride() == 3 and s.offset() == 50 and s));
  CHECK((not++s and not s));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.StripeIterator",
                  "[DataStructures][Unit]") {
  size_t i = 0;
  for (StripeIterator s(Index<3>(3, 4, 5), 0); s; ++s) {
    CHECK(s.offset() == i);
    CHECK(s.stride() == 1);
    i += 3;
  }
  i = 0;
  for (StripeIterator s(Index<3>(3, 4, 5), 2); s; ++s) {
    CHECK(s.offset() == i);
    CHECK(s.stride() == 12);
    i++;
  }
  check_stripe_iterator_helper(StripeIterator(Index<3>(3, 4, 5), 1));
}
