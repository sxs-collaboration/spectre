// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.KroneckerDelta",
                  "[DataStructures][Unit]") {
  static_assert(kdelta1.dim == 1, "kdelta1 dimension should be 1");
  static_assert(kdelta2.dim == 2, "kdelta2 dimension should be 2");
  static_assert(kdelta3.dim == 3, "kdelta3 dimension should be 3");
  static_assert(kdelta4.dim == 4, "kdelta4 dimension should be 4");

  // dimension 1
  const auto kdeltaIj = kdelta1(ti::I, ti::j);
  CHECK(kdeltaIj.get({{0}}) == 1.0);

  // dimension 2
  const auto kdeltakJ = kdelta2(ti::k, ti::J);
  CHECK(kdeltakJ.get({{0, 0}}) == 1.0);
  CHECK(kdeltakJ.get({{1, 0}}) == 0.0);
  CHECK(kdeltakJ.get({{0, 1}}) == 0.0);
  CHECK(kdeltakJ.get({{1, 1}}) == 1.0);

  // dimension 3 contracted
  const auto kdeltaJj = kdelta3(ti::J, ti::j);
  CHECK(kdeltaJj.get({{}}) == 3.0);

  // dimension 4, spacetime
  const auto kdeltaAb = kdelta4(ti::A, ti::b);
  for (size_t a = 0; a < 4; a++) {
    for (size_t b = 0; b < 4; b++) {
      if (a == b) {
        CHECK(kdeltaAb.get({{a, b}}) == 1.0);
      } else {
        CHECK(kdeltaAb.get({{a, b}}) == 0.0);
      }
    }
  }
}
