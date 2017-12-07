// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <catch.hpp>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_definite_integral_1d(const Index<1>& extents) {
  const size_t num_pts_in_x = extents[0];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  DataVector integrand(num_pts_in_x);
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t s = 0; s < integrand.size(); ++s) {
      integrand[s] = pow(x[s], a);
    }
    if (0 == a % 2) {
      CHECK(2.0 / (a + 1.0) ==
            approx(Basis::lgl::definite_integral(integrand, extents)));
    } else {
      CHECK(0.0 == approx(Basis::lgl::definite_integral(integrand, extents)));
    }
  }
}

void test_definite_integral_2d(const Index<2>& extents) {
  const size_t num_pts_in_x = extents[0];
  const size_t num_pts_in_y = extents[1];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& y = Basis::lgl::collocation_points(num_pts_in_y);
  DataVector integrand(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (IndexIterator<2> index_it(extents); index_it; ++index_it) {
        integrand[index_it.offset()] =
            pow(x[index_it()[0]], a) * pow(y[index_it()[1]], b);
      }
      if (0 == a % 2 and 0 == b % 2) {
        CHECK(4.0 / ((a + 1.0) * (b + 1.0)) ==
              approx(Basis::lgl::definite_integral(integrand, extents)));
      } else {
        CHECK(0.0 == approx(Basis::lgl::definite_integral(integrand, extents)));
      }
    }
  }
}

void test_definite_integral_3d(const Index<3>& extents) {
  const size_t num_pts_in_x = extents[0];
  const size_t num_pts_in_y = extents[1];
  const size_t num_pts_in_z = extents[2];
  const DataVector& x = Basis::lgl::collocation_points(num_pts_in_x);
  const DataVector& y = Basis::lgl::collocation_points(num_pts_in_y);
  const DataVector& z = Basis::lgl::collocation_points(num_pts_in_z);
  DataVector integrand(extents.product());
  for (size_t a = 0; a < num_pts_in_x; ++a) {
    for (size_t b = 0; b < num_pts_in_y; ++b) {
      for (size_t c = 0; c < num_pts_in_z; ++c) {
        for (IndexIterator<3> index_it(extents); index_it; ++index_it) {
          integrand[index_it.offset()] = pow(x[index_it()[0]], a) *
                                         pow(y[index_it()[1]], b) *
                                         pow(z[index_it()[2]], c);
        }
        if (0 == a % 2 and 0 == b % 2 and 0 == c % 2) {
          CHECK(8.0 / ((a + 1.0) * (b + 1.0) * (c + 1.0)) ==
                approx(Basis::lgl::definite_integral(integrand, extents)));
        } else {
          CHECK(0.0 ==
                approx(Basis::lgl::definite_integral(integrand, extents)));
        }
      }
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.DefiniteIntegral",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
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
