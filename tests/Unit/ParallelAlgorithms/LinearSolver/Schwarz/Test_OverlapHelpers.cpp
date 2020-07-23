// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "ErrorHandling/Error.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"

namespace LinearSolver::Schwarz {

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.OverlapHelpers",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  {
    INFO("Overlap extents");
    CHECK(overlap_extent(3, 0) == 0);
    CHECK(overlap_extent(3, 1) == 1);
    CHECK(overlap_extent(3, 2) == 2);
    CHECK(overlap_extent(3, 3) == 2);
    CHECK(overlap_extent(3, 4) == 2);
    CHECK(overlap_extent(0, 0) == 0);
  }
  {
    INFO("Overlap num_points");
    CHECK(overlap_num_points(Index<1>{{{3}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 1, 0) == 1);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 2, 0) == 2);
    CHECK(overlap_num_points(Index<1>{{{3}}}, 3, 0) == 3);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 1, 0) == 3);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 2, 0) == 6);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 0, 1) == 0);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 1, 1) == 2);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 2, 1) == 4);
    CHECK(overlap_num_points(Index<2>{{{2, 3}}}, 3, 1) == 6);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 0) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 0) == 12);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 0) == 24);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 1) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 1) == 8);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 1) == 16);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 3, 1) == 24);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 0, 2) == 0);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 1, 2) == 6);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 2, 2) == 12);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 3, 2) == 18);
    CHECK(overlap_num_points(Index<3>{{{2, 3, 4}}}, 4, 2) == 24);
  }
  {
    INFO("Overlap width");
    DataVector coords{-1., -0.8, 0., 0.8, 1.};
    CHECK(overlap_width(0, coords) == approx(0.));
    CHECK(overlap_width(1, coords) == approx(0.2));
    CHECK(overlap_width(2, coords) == approx(1.));
    CHECK(overlap_width(3, coords) == approx(1.8));
    CHECK(overlap_width(4, coords) == approx(2.));
  }
}

// [[OutputRegex, Overlap extent '4' exceeds volume extents]]
[[noreturn]] SPECTRE_TEST_CASE(
    "Unit.ParallelSchwarz.OverlapHelpers.AssertOverlapNumPoints",
    "[Unit][ParallelAlgorithms][LinearSolver]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  overlap_num_points(Index<1>{{{3}}}, 4, 0);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Invalid dimension '1' in 1D]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.ParallelSchwarz.OverlapHelpers.AssertDim",
                               "[Unit][ParallelAlgorithms][LinearSolver]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  overlap_num_points(Index<1>{{{3}}}, 0, 1);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

}  // namespace LinearSolver::Schwarz
