
#include <catch.hpp>
#include <cmath>

#include "Numerical/Spectral/DefiniteIntegral.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "DataStructures/Mesh.hpp"
#include "Numerical/Spectral/LegendreGaussLobatto.hpp"

void TestDefiniteIntegral1D(const Index<1>& N) {
  const size_t N0 = N[0];
  const DataVector& x = Basis::lgl::collocation_points(N0);
  DataVector f(N0);
  for (size_t a = 0; a < N[0]; ++a) {
    for (size_t s = 0; s < f.size(); ++s) {
      f[s] = pow(x[s], a);
    }
    if (0 == a % 2) {
      CHECK(2.0 / (a + 1) == Approx(definite_integral(f, N)));
    } else {
      CHECK(0.0 == Approx(definite_integral(f, N)));
    }
  }
}

void TestDefiniteIntegral2D(const Index<2>& N) {
  Mesh<2> extents(N);
  const size_t N0 = N[0];
  const size_t N1 = N[1];
  const DataVector& x0 = Basis::lgl::collocation_points(N0);
  const DataVector& x1 = Basis::lgl::collocation_points(N1);
  DataVector f(extents.product());
  for (size_t a = 0; a < N[0]; ++a) {
    for (size_t b = 0; b < N[1]; ++b) {
      for (IndexIterator<2> I(N); I; ++I) {
        f[I.offset()] = pow(x0[I()[0]], a) * pow(x1[I()[1]], b);
      }
      if (0 == a % 2 && 0 == b % 2) {
        CHECK(4.0 / ((a + 1) * (b + 1)) == Approx(definite_integral(f, N)));
      } else {
        CHECK(0.0 == Approx(definite_integral(f, N)));
      }
    }
  }
}

void TestDefiniteIntegral3D(const Index<3>& N) {
  Mesh<3> extents(N);
  const size_t N0 = N[0];
  const size_t N1 = N[1];
  const size_t N2 = N[2];
  const DataVector& x0 = Basis::lgl::collocation_points(N0);
  const DataVector& x1 = Basis::lgl::collocation_points(N1);
  const DataVector& x2 = Basis::lgl::collocation_points(N2);
  DataVector f(extents.product());
  for (size_t a = N[0] / 2; a < N[0]; ++a) {
    for (size_t b = N[1] / 2; b < N[1]; ++b) {
      for (size_t c = N[2] / 2; c < N[2]; ++c) {
        for (IndexIterator<3> I(N); I; ++I) {
          f[I.offset()] =
              pow(x0[I()[0]], a) * pow(x1[I()[1]], b) * pow(x2[I()[2]], c);
        }
        if (0 == a % 2 && 0 == b % 2 && 0 == c % 2) {
          CHECK(8.0 / ((a + 1) * (b + 1) * (c + 1)) ==
                Approx(definite_integral(f, N)));
        } else {
          CHECK(0.0 == Approx(definite_integral(f, N)));
        }
      }
    }
  }
}

TEST_CASE("Unit.Functors.DefiniteIntegral", "[Functors][Unit]") {
  const size_t start_points = 3;
  const size_t end_points = 4;
  for (size_t n0 = start_points; n0 < end_points; ++n0) {
    Index<1> N1(n0);
    TestDefiniteIntegral1D(N1);
    for (size_t n1 = start_points; n1 < end_points; ++n1) {
      Index<2> N2(n0, n1);
      TestDefiniteIntegral2D(N2);
      for (size_t n2 = start_points; n2 < end_points; ++n2) {
        Index<3> N3(n0, n1, n2);
        TestDefiniteIntegral3D(N3);
      }
    }
  }
}
