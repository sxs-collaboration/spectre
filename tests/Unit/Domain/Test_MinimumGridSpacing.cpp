// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/MinimumGridSpacing.hpp"

// IWYU pragma: no_include "Utilities/Array.hpp"

namespace {
template <size_t Dim, typename Frame>
void check(const Matrix& transform, const double squeeze_factor) noexcept {
  CAPTURE(transform);
  const double large_segment = 3.0;
  const double small_segment = 1.7;

  Index<Dim> extents{};
  Index<Dim> cells{};
  for (size_t d = 0; d < Dim; ++d) {
    extents[d] = 4 + d;
    cells[d] = 3 + d;
  }
  for (IndexIterator<Dim> small_cell(cells); small_cell; ++small_cell) {
    // Fill coordinates
    tnsr::I<DataVector, Dim, Frame> untransformed_coordinates(
        extents.product());
    for (IndexIterator<Dim> point(extents); point; ++point) {
      for (size_t d = 0; d < Dim; ++d) {
        double& coordinate =
            untransformed_coordinates.get(d)[point.collapsed_index()];
        coordinate = large_segment * (*point)[d];
        if ((*point)[d] > (*small_cell)[d]) {
          coordinate -= large_segment - small_segment;
        }
      }
    }
    tnsr::I<DataVector, Dim, Frame> coordinates(extents.product(), 0.);
    for (size_t d = 0; d < Dim; ++d) {
      for (size_t d2 = 0; d2 < Dim; ++d2) {
        coordinates.get(d) +=
            transform(d, d2) * untransformed_coordinates.get(d2);
      }
    }

    CHECK(minimum_grid_spacing(extents, coordinates) ==
          approx(small_segment * squeeze_factor));
  }
}

template <typename Frame>
void check_frame() noexcept {
  check<1, Frame>(Matrix{{1.}}, 1.);
  check<1, Frame>(Matrix{{-1.}}, 1.);
  check<1, Frame>(Matrix{{0.1}}, 0.1);
  check<1, Frame>(Matrix{{-0.1}}, 0.1);

  check<2, Frame>(Matrix{{1.0, 0.0}, {0.0, 1.0}}, 1.0);
  check<2, Frame>(Matrix{{1.0, 0.0}, {0.0, -1.0}}, 1.0);
  check<2, Frame>(Matrix{{0.0, 1.0}, {1.0, 0.0}}, 1.0);
  check<2, Frame>(Matrix{{0.5, 0.0}, {0.0, 1.0}}, 0.5);
  check<2, Frame>(Matrix{{1.0, 0.0}, {0.0, 0.5}}, 0.5);
  // Rotated grid
  check<2, Frame>(Matrix{{1.0, 1.0}, {1.0, -1.0}}, sqrt(2.0));
  // Triangular grid
  check<2, Frame>(Matrix{{1.0, 0.5}, {0.0, sqrt(0.75)}}, 1.0);
  // Squashed grid
  check<2, Frame>(Matrix{{1.0, 1.0}, {0.0, 0.2}}, 0.2);
  check<2, Frame>(Matrix{{1.0, -1.0}, {0.0, 0.2}}, 0.2);

  check<3, Frame>(Matrix{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  1.0);
  check<3, Frame>(Matrix{{-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  1.0);
  check<3, Frame>(Matrix{{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}},
                  1.0);
  check<3, Frame>(Matrix{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, -1.0}},
                  1.0);
  check<3, Frame>(Matrix{{0.5, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}},
                  0.5);
  check<3, Frame>(Matrix{{1.0, 0.0, 0.0}, {0.0, 0.5, 1.0}, {0.0, 0.0, 1.0}},
                  0.5);
  check<3, Frame>(Matrix{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 0.5}},
                  0.5);
  check<3, Frame>(
      Matrix{{1.0, 1.0, -1.0}, {1.0, -1.0, 1.0}, {-1.0, 1.0, 1.0}}, sqrt(3.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.MinimumGridSpacing", "[Domain][Unit]") {
  check_frame<Frame::Grid>();
  check_frame<Frame::Inertial>();
}
