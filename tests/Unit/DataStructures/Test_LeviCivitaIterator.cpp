// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.LeviCivitaIterator",
                  "[DataStructures][Unit]") {
  // Test 1D
  std::array<int, 1> signs_1d = {{1}};
  std::array<std::array<size_t, 1>, 1> indexes_1d = {
      {std::array<size_t, 1>{{0}}}};

  size_t i = 0;
  for (LeviCivitaIterator<1> it; it; ++it) {
    CHECK(it() == gsl::at(indexes_1d, i));
    CHECK(it.sign() == gsl::at(signs_1d, i));
    ++i;
  }
  CHECK(i == 1);

  // Test 2D
  std::array<int, 2> signs_2d = {{1, -1}};
  std::array<std::array<size_t, 2>, 2> indexes_2d = {
      {std::array<size_t, 2>{{0, 1}}, std::array<size_t, 2>{{1, 0}}}};

  i = 0;
  for (LeviCivitaIterator<2> it; it; ++it) {
    CHECK(it() == gsl::at(indexes_2d, i));
    CHECK(it.sign() == gsl::at(signs_2d, i));
    ++i;
  }
  CHECK(i == 2);

  // Test 3D
  std::array<int, 6> signs_3d = {{1, -1, -1, 1, 1, -1}};
  std::array<std::array<size_t, 3>, 6> indexes_3d = {
      {std::array<size_t, 3>{{0, 1, 2}}, std::array<size_t, 3>{{0, 2, 1}},
       std::array<size_t, 3>{{1, 0, 2}}, std::array<size_t, 3>{{1, 2, 0}},
       std::array<size_t, 3>{{2, 0, 1}}, std::array<size_t, 3>{{2, 1, 0}}}};

  i = 0;
  for (LeviCivitaIterator<3> it; it; ++it) {
    CHECK(it() == gsl::at(indexes_3d, i));
    CHECK(it.sign() == gsl::at(signs_3d, i));
    ++i;
  }
  CHECK(i == 6);

  // Test 4D
  std::array<int, 24> signs_4d = {{1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1,
                                   1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1}};
  std::array<std::array<size_t, 4>, 24> indexes_4d = {
      {std::array<size_t, 4>{{0, 1, 2, 3}},
       std::array<size_t, 4>{{0, 1, 3, 2}},
       std::array<size_t, 4>{{0, 2, 1, 3}},
       std::array<size_t, 4>{{0, 2, 3, 1}},
       std::array<size_t, 4>{{0, 3, 1, 2}},
       std::array<size_t, 4>{{0, 3, 2, 1}},
       std::array<size_t, 4>{{1, 0, 2, 3}},
       std::array<size_t, 4>{{1, 0, 3, 2}},
       std::array<size_t, 4>{{1, 2, 0, 3}},
       std::array<size_t, 4>{{1, 2, 3, 0}},
       std::array<size_t, 4>{{1, 3, 0, 2}},
       std::array<size_t, 4>{{1, 3, 2, 0}},
       std::array<size_t, 4>{{2, 0, 1, 3}},
       std::array<size_t, 4>{{2, 0, 3, 1}},
       std::array<size_t, 4>{{2, 1, 0, 3}},
       std::array<size_t, 4>{{2, 1, 3, 0}},
       std::array<size_t, 4>{{2, 3, 0, 1}},
       std::array<size_t, 4>{{2, 3, 1, 0}},
       std::array<size_t, 4>{{3, 0, 1, 2}},
       std::array<size_t, 4>{{3, 0, 2, 1}},
       std::array<size_t, 4>{{3, 1, 0, 2}},
       std::array<size_t, 4>{{3, 1, 2, 0}},
       std::array<size_t, 4>{{3, 2, 0, 1}},
       std::array<size_t, 4>{{3, 2, 1, 0}}}};

  i = 0;
  for (LeviCivitaIterator<4> it; it; ++it) {
    CHECK(it() == gsl::at(indexes_4d, i));
    CHECK(it.sign() == gsl::at(signs_4d, i));
    ++i;
  }
  CHECK(i == 24);

  // Demonstrate using the iterator to compute a cross product
  /// [levi_civita_iterator_example]
  const std::array<double, 3> vector_a = {{2.0, 3.0, 4.0}};
  const std::array<double, 3> vector_b = {{7.0, 6.0, 5.0}};
  const std::array<double, 3> a_cross_b_expected = {{-9.0, 18.0, -9.0}};

  std::array<double, 3> a_cross_b = {{0.0, 0.0, 0.0}};
  for (LeviCivitaIterator<3> it; it; ++it) {
    gsl::at(a_cross_b, it[0]) +=
        it.sign() * gsl::at(vector_a, it[1]) * gsl::at(vector_b, it[2]);
  }
  CHECK_ITERABLE_APPROX(a_cross_b, a_cross_b_expected);
  /// [levi_civita_iterator_example]
}
