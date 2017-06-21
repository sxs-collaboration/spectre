
#include <catch.hpp>
#include <cmath>

#include "Numerical/Spectral/DefiniteIntegral.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Mesh.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"

namespace {
void test_definite_integral_1D(const Index<1>& index_1d) {
  Approx approx = Approx::custom().epsilon(1e-5);
  const size_t num_pts_in_x = index_1d[0];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  DataVector f(num_pts_in_x);
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t s = 0; s < f.size(); ++s) {
      f[s] = pow(x[s], a);
    }
    if (0 == a % 2) {
      CHECK(2.0 / (a + 1.0) == approx(definite_integral(f, index_1d)));
    } else {
      CHECK(0.0 == approx(definite_integral(f, index_1d)));
    }
  }
}

void test_definite_integral_2D(const Index<2>& index_2d) {
  Approx approx = Approx::custom().epsilon(1e-5);
  Mesh<2> extents(index_2d);
  const size_t num_pts_in_x = index_2d[0];
  const size_t num_pts_in_y = index_2d[1];
  const DataVector& x0 = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& x1 = Basis::lgl::collocation_points(num_pts_in_y);
  DataVector f(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (IndexIterator<2> I(index_2d); I; ++I) {
        f[I.offset()] = pow(x0[I()[0]], a) * pow(x1[I()[1]], b);
      }
      if (0 == a % 2 and 0 == b % 2) {
        CHECK(4.0 / ((a + 1.0) * (b + 1.0)) ==
              approx(definite_integral(f, index_2d)));
      } else {
        CHECK(0.0 == approx(definite_integral(f, index_2d)));
      }
    }
  }
}

void test_definite_integral_3D(const Index<3>& index_3d) {
  Approx approx = Approx::custom().epsilon(1e-5);
  Mesh<3> extents(index_3d);
  const size_t num_pts_in_x = index_3d[0];
  const size_t num_pts_in_y = index_3d[1];
  const size_t num_pts_in_z = index_3d[2];
  const DataVector& x0 = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& x1 = Basis::lgl::collocation_points(num_pts_in_y);
  const DataVector& x2 = Basis::lgl::collocation_points(num_pts_in_z);
  DataVector f(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (size_t c = 0; c < num_pts_in_z; ++c) {
        for (IndexIterator<3> I(index_3d); I; ++I) {
          f[I.offset()] =
              pow(x0[I()[0]], a) * pow(x1[I()[1]], b) * pow(x2[I()[2]], c);
        }
        if (0 == a % 2 and 0 == b % 2 and 0 == c % 2) {
          CHECK(8.0 / ((a + 1.0) * (b + 1.0) * (c + 1.0)) ==
                approx(definite_integral(f, index_3d)));
        } else {
          CHECK(0.0 == approx(definite_integral(f, index_3d)));
        }
      }
    }
  }
}
}  // namespace

TEST_CASE("Unit.Numerical.Spectral.DefiniteIntegral", "[Functors][Unit]") {
  const size_t min_extents = 2;
  for (size_t n0 = min_extents; n0 <= Basis::lgl::maximum_number_of_pts; ++n0) {
    test_definite_integral_1D(Index<1>(n0));
  }
  for (size_t n0 = min_extents; n0 <= Basis::lgl::maximum_number_of_pts; ++n0) {
    for (size_t n1 = min_extents; n1 <= Basis::lgl::maximum_number_of_pts - 1;
         ++n1) {
      test_definite_integral_2D(Index<2>(n0, n1));
    }
  }
  for (size_t n0 = min_extents; n0 <= 6; ++n0) {
    for (size_t n1 = min_extents; n1 <= 7; ++n1) {
      for (size_t n2 = min_extents; n2 <= 8; ++n2) {
        test_definite_integral_3D(Index<3>(n0, n1, n2));
      }
    }
  }
}
