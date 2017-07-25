// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataVector.hpp"
#include "Numerical/LinearAlgebra/Determinant.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.LinearAlgebra.Determinant",
                  "[LinearAlgebra][Numerical][Unit]") {
  Approx approx = Approx::custom().epsilon(1e-15);

  // Test determinant function on general (no symmetry) matrices:
  // * use rank-2 Tensor in 1-4 dimensions, i.e. 1x1, 2x2, 3x3, 4x4 matrices.
  // * use Tensor<double, ...>, i.e. data at single spatial point.
  SECTION("Test general Tensor<double,...> matrices") {
    {
      const tnsr::ij<double, 1, Frame::Grid> matrix(1.3);
      const auto det = determinant(matrix);
      CHECK(1.3 == approx(det.get()));
    }

    {
      tnsr::ij<double, 2, Frame::Grid> matrix;
      matrix.get<0, 0>() = 2.1;
      matrix.get<1, 0>() = 4.5;
      matrix.get<0, 1>() = -18.3;
      matrix.get<1, 1>() = -10.9;
      const auto det = determinant(matrix);
      CHECK(59.46 == approx(det.get()));
    }

    {
      tnsr::ij<double, 3, Frame::Grid> matrix;
      matrix.get<0, 0>() = 1.1;
      matrix.get<1, 0>() = -10.4;
      matrix.get<2, 0>() = -4.5;
      matrix.get<0, 1>() = 3.2;
      matrix.get<1, 1>() = 15.4;
      matrix.get<2, 1>() = -19.2;
      matrix.get<0, 2>() = 4.3;
      matrix.get<1, 2>() = 16.8;
      matrix.get<2, 2>() = 2.0;
      const auto det = determinant(matrix);
      CHECK(1369.95 == approx(det.get()));
    }

    {
      tnsr::ij<double, 4, Frame::Grid> matrix;
      matrix.get<0, 0>() = 1.1;
      matrix.get<1, 0>() = -10.4;
      matrix.get<2, 0>() = -4.5;
      matrix.get<3, 0>() = 3.2;
      matrix.get<0, 1>() = 15.4;
      matrix.get<1, 1>() = -19.2;
      matrix.get<2, 1>() = 4.3;
      matrix.get<3, 1>() = 16.8;
      matrix.get<0, 2>() = 2.0;
      matrix.get<1, 2>() = 3.1;
      matrix.get<2, 2>() = 4.2;
      matrix.get<3, 2>() = -0.2;
      matrix.get<0, 3>() = -3.2;
      matrix.get<1, 3>() = 2.6;
      matrix.get<2, 3>() = 1.5;
      matrix.get<3, 3>() = 2.8;
      const auto det = determinant(matrix);
      CHECK(1049.7072 == approx(det.get()));
    }
  }

  // Test determinant function on symmetric matrices:
  // * use rank-2 Tensor in 2-4 dimensions (symmetry meaningless for dim==1).
  // * use Tensor<double, ...>, i.e. data at single spatial point.
  SECTION("Test symmetric Tensor<double,...> matrices") {
    {
      tnsr::ii<double, 2, Frame::Grid> matrix;
      matrix.get<0, 0>() = 2.1;
      matrix.get<1, 0>() = 4.5;
      matrix.get<1, 1>() = -10.9;
      const auto det = determinant(matrix);
      CHECK(-43.14 == approx(det.get()));
    }

    {
      tnsr::ii<double, 3, Frame::Grid> matrix;
      matrix.get<0, 0>() = 1.1;
      matrix.get<1, 0>() = -10.4;
      matrix.get<2, 0>() = -4.5;
      matrix.get<1, 1>() = 3.2;
      matrix.get<2, 1>() = 15.4;
      matrix.get<2, 2>() = -19.2;
      const auto det = determinant(matrix);
      CHECK(3124.852 == approx(det.get()));
    }

    {
      tnsr::ii<double, 4, Frame::Grid> matrix;
      matrix.get<0, 0>() = 1.1;
      matrix.get<1, 0>() = -10.4;
      matrix.get<2, 0>() = -4.5;
      matrix.get<3, 0>() = 3.2;
      matrix.get<1, 1>() = 15.4;
      matrix.get<2, 1>() = -19.2;
      matrix.get<3, 1>() = 4.3;
      matrix.get<2, 2>() = 16.8;
      matrix.get<3, 2>() = 2.0;
      matrix.get<3, 3>() = 3.1;
      const auto det = determinant(matrix);
      CHECK(-22819.6093 == approx(det.get()));
    }
  }

  // Test determinant function for Tensors of different (not double) types:
  // * use rank-2 Tensor in 2 dimensions, i.e. a 2x2 matrix.
  // * use Tensor<T,...> for T in {int, DataVector}.
  SECTION("Test Tensors of different types") {
    {
      tnsr::ij<int, 2, Frame::Grid> matrix;
      matrix.get<0, 0>() = 9;
      matrix.get<1, 0>() = 1;
      matrix.get<0, 1>() = -3;
      matrix.get<1, 1>() = -4;
      const auto det = determinant(matrix);
      CHECK(-33 == det.get());
    }

    {
      tnsr::ij<DataVector, 2, Frame::Grid> matrix;
      matrix.get<0, 0>() = DataVector({6.0, 5.9, 9.8, 6.4});
      matrix.get<1, 0>() = DataVector({6.1, 0.3, 2.4, 5.7});
      matrix.get<0, 1>() = DataVector({4.2, 7.1, 1.1, 6.5});
      matrix.get<1, 1>() = DataVector({7.2, 8.4, 6.1, 3.7});
      const auto det = determinant(matrix);
      CHECK(17.58 == approx(det.get()[0]));
      CHECK(47.43 == approx(det.get()[1]));
      CHECK(57.14 == approx(det.get()[2]));
      CHECK(-13.37 == approx(det.get()[3]));
    }
  }
}
