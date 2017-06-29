// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <catch.hpp>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Mesh.hpp"
#include "Numerical/Spectral/DefiniteIntegral.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"

namespace {
void test_definite_integral_1d(const Index<1>& index_1d) {
  Approx approx = Approx::custom().epsilon(1e-15);
  const size_t num_pts_in_x = index_1d[0];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  Scalar<DataVector> integrand(num_pts_in_x);
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t s = 0; s < integrand.begin()->size(); ++s) {
      integrand.get()[s] = pow(x[s], a);
    }
    if (0 == a % 2) {
      CHECK(2.0 / (a + 1.0) ==
            approx(Basis::lgl::definite_integral(integrand, index_1d)));
    } else {
      CHECK(0.0 == approx(Basis::lgl::definite_integral(integrand, index_1d)));
    }
  }
}

void test_definite_integral_2d(const Index<2>& index_2d) {
  Approx approx = Approx::custom().epsilon(1e-15);
  Mesh<2> extents(index_2d);
  const size_t num_pts_in_x = index_2d[0];
  const size_t num_pts_in_y = index_2d[1];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& y = Basis::lgl::collocation_points(num_pts_in_y);
  Scalar<DataVector> integrand(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (IndexIterator<2> index_it(index_2d); index_it; ++index_it) {
        integrand.get()[index_it.offset()] =
            pow(x[index_it()[0]], a) * pow(y[index_it()[1]], b);
      }
      if (0 == a % 2 and 0 == b % 2) {
        CHECK(4.0 / ((a + 1.0) * (b + 1.0)) ==
              approx(Basis::lgl::definite_integral(integrand, index_2d)));
      } else {
        CHECK(0.0 ==
              approx(Basis::lgl::definite_integral(integrand, index_2d)));
      }
    }
  }
}

void test_definite_integral_3d(const Index<3>& index_3d) {
  Approx approx = Approx::custom().epsilon(1e-15);
  Mesh<3> extents(index_3d);
  const size_t num_pts_in_x = index_3d[0];
  const size_t num_pts_in_y = index_3d[1];
  const size_t num_pts_in_z = index_3d[2];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& y = Basis::lgl::collocation_points(num_pts_in_y);
  const DataVector& z = Basis::lgl::collocation_points(num_pts_in_z);
  Scalar<DataVector> integrand(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (size_t c = 0; c < num_pts_in_z; ++c) {
        for (IndexIterator<3> index_it(index_3d); index_it; ++index_it) {
          integrand.get()[index_it.offset()] = pow(x[index_it()[0]], a) *
                                               pow(y[index_it()[1]], b) *
                                               pow(z[index_it()[2]], c);
        }
        if (0 == a % 2 and 0 == b % 2 and 0 == c % 2) {
          CHECK(8.0 / ((a + 1.0) * (b + 1.0) * (c + 1.0)) ==
                approx(Basis::lgl::definite_integral(integrand, index_3d)));
        } else {
          CHECK(0.0 ==
                approx(Basis::lgl::definite_integral(integrand, index_3d)));
        }
      }
    }
  }
}
}  // namespace

TEST_CASE("Unit.Numerical.Spectral.DefiniteIntegral", "[Functors][Unit]") {
  const size_t min_extents = 2;
  for (size_t n0 = min_extents; n0 <= Basis::lgl::maximum_number_of_pts; ++n0) {
    test_definite_integral_1d(Index<1>(n0));
  }
  for (size_t n0 = min_extents; n0 <= Basis::lgl::maximum_number_of_pts; ++n0) {
    for (size_t n1 = min_extents; n1 <= Basis::lgl::maximum_number_of_pts - 1;
         ++n1) {
      test_definite_integral_2d(Index<2>(n0, n1));
    }
  }
  for (size_t n0 = min_extents;
       n0 <= std::min(6_st, Basis::lgl::maximum_number_of_pts); ++n0) {
    for (size_t n1 = min_extents;
         n1 <= std::min(7_st, Basis::lgl::maximum_number_of_pts); ++n1) {
      for (size_t n2 = min_extents;
           n2 <= std::min(8_st, Basis::lgl::maximum_number_of_pts); ++n2) {
        test_definite_integral_3d(Index<3>(n0, n1, n2));
      }
    }
  }
}
