// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Index.hpp"
#include "DataStructures/SliceIterator.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void check_slice_iterator_helper(SliceIterator si) {
  CHECK((si.slice_offset() == 0 and si.volume_offset() == 0 and si and ++si));
  CHECK((si.slice_offset() == 1 and si.volume_offset() == 1 and si and ++si));
  CHECK((si.slice_offset() == 2 and si.volume_offset() == 2 and si and ++si));
  CHECK((si.slice_offset() == 3 and si.volume_offset() == 12 and si and ++si));
  CHECK((si.slice_offset() == 4 and si.volume_offset() == 13 and si and ++si));
  CHECK((si.slice_offset() == 5 and si.volume_offset() == 14 and si and ++si));
  CHECK((si.slice_offset() == 6 and si.volume_offset() == 24 and si and ++si));
  CHECK((si.slice_offset() == 7 and si.volume_offset() == 25 and si and ++si));
  CHECK((si.slice_offset() == 8 and si.volume_offset() == 26 and si and ++si));
  CHECK((si.slice_offset() == 9 and si.volume_offset() == 36 and si and ++si));
  CHECK((si.slice_offset() == 10 and si.volume_offset() == 37 and si and ++si));
  CHECK((si.slice_offset() == 11 and si.volume_offset() == 38 and si and ++si));
  CHECK((si.slice_offset() == 12 and si.volume_offset() == 48 and si and ++si));
  CHECK((si.slice_offset() == 13 and si.volume_offset() == 49 and si and ++si));
  CHECK((si.slice_offset() == 14 and si.volume_offset() == 50 and si));
  CHECK((not++si and not si));
}

template <size_t Dim>
void check_slice_and_volume_indices(const Index<Dim>& extents) noexcept {
  const auto t = volume_and_slice_indices(extents);
  const auto& indices = t.second;
  for (size_t d = 0; d < Dim; ++d) {
    size_t index = 0;
    for (SliceIterator sli_lower(extents, d, 0),
         sli_upper(extents, d, extents[d] - 1);
         sli_lower and sli_upper;
         (void)++sli_lower, (void)++sli_upper, (void)++index) {
      CHECK(gsl::at(gsl::at(indices, d).first, index).first ==
            sli_lower.volume_offset());
      CHECK(gsl::at(gsl::at(indices, d).first, index).second ==
            sli_lower.slice_offset());

      CHECK(gsl::at(gsl::at(indices, d).second, index).first ==
            sli_upper.volume_offset());
      CHECK(gsl::at(gsl::at(indices, d).second, index).second ==
            sli_upper.slice_offset());
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.SliceIterator",
                  "[DataStructures][Unit]") {
  SECTION("SliceIterator class") {
    size_t i = 0;
    for (SliceIterator si(Index<3>(3, 4, 5), 2, 0); si; ++si) {
      CHECK(i == si.slice_offset());
      CHECK(i == si.volume_offset());
      i++;
    }
    SliceIterator slice_iter(Index<3>(3, 4, 5), 1, 0);
    check_slice_iterator_helper(slice_iter);
    slice_iter.reset();
    check_slice_iterator_helper(slice_iter);
  }
  SECTION("volume_and_slice_indices function") {
    check_slice_and_volume_indices(Index<1>{2});
    check_slice_and_volume_indices(Index<2>{2, 2});
    check_slice_and_volume_indices(Index<2>{5, 4});
    check_slice_and_volume_indices(Index<2>{3, 4});
    check_slice_and_volume_indices(Index<3>{2, 2, 2});
    check_slice_and_volume_indices(Index<3>{6, 4, 3});
    check_slice_and_volume_indices(Index<3>{6, 4, 5});
    check_slice_and_volume_indices(Index<3>{3, 4, 5});
  }
}
