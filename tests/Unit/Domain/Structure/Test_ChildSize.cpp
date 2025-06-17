// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/Spectral/SegmentSize.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Structure.ChildSize", "[Domain][Unit]") {
  CHECK(child_size({0, 0}, {0, 0}) == Spectral::SegmentSize::Full);
  CHECK(child_size({1, 0}, {0, 0}) == Spectral::SegmentSize::LowerHalf);
  CHECK(child_size({1, 1}, {0, 0}) == Spectral::SegmentSize::UpperHalf);
  CHECK(child_size({1, 1}, {1, 1}) == Spectral::SegmentSize::Full);
  CHECK(child_size<0>({}, {}).empty());
  CHECK(
      child_size<1>({{{2, 3}}}, {{{1, 1}}}) ==
      std::array<Spectral::SegmentSize, 1>{{Spectral::SegmentSize::UpperHalf}});
  CHECK(child_size<2>({{{0, 0}, {1, 0}}}, {{{0, 0}, {0, 0}}}) ==
        std::array<Spectral::SegmentSize, 2>{
            {Spectral::SegmentSize::Full, Spectral::SegmentSize::LowerHalf}});
  CHECK(child_size<3>({{{1, 1}, {1, 1}, {2, 2}}}, {{{0, 0}, {1, 1}, {1, 1}}}) ==
        std::array<Spectral::SegmentSize, 3>{
            {Spectral::SegmentSize::UpperHalf, Spectral::SegmentSize::Full,
             Spectral::SegmentSize::LowerHalf}});

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH((child_size({1, 1}, {1, 0})),
                    Catch::Matchers::ContainsSubstring(
                        "Segment id 'L1I0' is not the parent of 'L1I1'."));
#endif
}

}  // namespace domain
