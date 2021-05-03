// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/Structure/ChildSize.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.Structure.ChildSize", "[Domain][Unit]") {
  CHECK(child_size({0, 0}, {0, 0}) == Spectral::ChildSize::Full);
  CHECK(child_size({1, 0}, {0, 0}) == Spectral::ChildSize::LowerHalf);
  CHECK(child_size({1, 1}, {0, 0}) == Spectral::ChildSize::UpperHalf);
  CHECK(child_size({1, 1}, {1, 1}) == Spectral::ChildSize::Full);
  CHECK(child_size<1>({{{2, 3}}}, {{{1, 1}}}) ==
        std::array<Spectral::ChildSize, 1>{{Spectral::ChildSize::UpperHalf}});
  CHECK(child_size<2>({{{0, 0}, {1, 0}}}, {{{0, 0}, {0, 0}}}) ==
        std::array<Spectral::ChildSize, 2>{
            {Spectral::ChildSize::Full, Spectral::ChildSize::LowerHalf}});
  CHECK(child_size<3>({{{1, 1}, {1, 1}, {2, 2}}}, {{{0, 0}, {1, 1}, {1, 1}}}) ==
        std::array<Spectral::ChildSize, 3>{{Spectral::ChildSize::UpperHalf,
                                            Spectral::ChildSize::Full,
                                            Spectral::ChildSize::LowerHalf}});
}

// [[OutputRegex, Segment id 'L1I0' is not the parent of 'L1I1'.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Structure.ChildSize.Assert",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  child_size({1, 1}, {1, 0});
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

}  // namespace domain
