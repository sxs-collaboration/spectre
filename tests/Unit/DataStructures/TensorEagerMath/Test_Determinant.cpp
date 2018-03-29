// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.EagerMath.Determinant",
                  "[DataStructures][Unit]") {
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
      tnsr::ij<double, 2, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 2.1;
      get<0, 1>(matrix) = 4.5;
      get<1, 0>(matrix) = -18.3;
      get<1, 1>(matrix) = -10.9;
      const auto det = determinant(matrix);
      CHECK(59.46 == approx(det.get()));
    }

    {
      tnsr::ij<double, 3, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 1.1;
      get<0, 1>(matrix) = -10.4;
      get<0, 2>(matrix) = -4.5;
      get<1, 0>(matrix) = 3.2;
      get<1, 1>(matrix) = 15.4;
      get<1, 2>(matrix) = -19.2;
      get<2, 0>(matrix) = 4.3;
      get<2, 1>(matrix) = 16.8;
      get<2, 2>(matrix) = 2.0;
      const auto det = determinant(matrix);
      CHECK(1369.95 == approx(det.get()));
    }

    {
      tnsr::ij<double, 4, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 1.1;
      get<0, 1>(matrix) = -10.4;
      get<0, 2>(matrix) = -4.5;
      get<0, 3>(matrix) = 3.2;
      get<1, 0>(matrix) = 15.4;
      get<1, 1>(matrix) = -19.2;
      get<1, 2>(matrix) = 4.3;
      get<1, 3>(matrix) = 16.8;
      get<2, 0>(matrix) = 2.0;
      get<2, 1>(matrix) = 3.1;
      get<2, 2>(matrix) = 4.2;
      get<2, 3>(matrix) = -0.2;
      get<3, 0>(matrix) = -3.2;
      get<3, 1>(matrix) = 2.6;
      get<3, 2>(matrix) = 1.5;
      get<3, 3>(matrix) = 2.8;
      const auto det = determinant(matrix);
      CHECK(1049.7072 == approx(det.get()));
    }
  }

  // Test determinant function on symmetric matrices:
  // * use rank-2 Tensor in 2-4 dimensions (symmetry meaningless for dim==1).
  // * use Tensor<double, ...>, i.e. data at single spatial point.
  SECTION("Test symmetric Tensor<double,...> matrices") {
    {
      tnsr::ii<double, 2, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 2.1;
      get<0, 1>(matrix) = 4.5;
      get<1, 1>(matrix) = -10.9;
      const auto det = determinant(matrix);
      CHECK(-43.14 == approx(det.get()));
    }

    {
      tnsr::ii<double, 3, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 1.1;
      get<0, 1>(matrix) = -10.4;
      get<0, 2>(matrix) = -4.5;
      get<1, 1>(matrix) = 3.2;
      get<1, 2>(matrix) = 15.4;
      get<2, 2>(matrix) = -19.2;
      const auto det = determinant(matrix);
      CHECK(3124.852 == approx(det.get()));
    }

    {
      tnsr::ii<double, 4, Frame::Grid> matrix{};
      get<0, 0>(matrix) = 1.1;
      get<0, 1>(matrix) = -10.4;
      get<0, 2>(matrix) = -4.5;
      get<0, 3>(matrix) = 3.2;
      get<1, 1>(matrix) = 15.4;
      get<1, 2>(matrix) = -19.2;
      get<1, 3>(matrix) = 4.3;
      get<2, 2>(matrix) = 16.8;
      get<2, 3>(matrix) = 2.0;
      get<3, 3>(matrix) = 3.1;
      const auto det = determinant(matrix);
      CHECK(-22819.6093 == approx(det.get()));
    }
  }

  SECTION("Test Tensors of DataVector") {
    tnsr::ij<DataVector, 2, Frame::Grid> matrix{};
    get<0, 0>(matrix) = DataVector({6.0, 5.9, 9.8, 6.4});
    get<0, 1>(matrix) = DataVector({6.1, 0.3, 2.4, 5.7});
    get<1, 0>(matrix) = DataVector({4.2, 7.1, 1.1, 6.5});
    get<1, 1>(matrix) = DataVector({7.2, 8.4, 6.1, 3.7});
    const auto det = determinant(matrix);
    CHECK(17.58 == approx(det.get()[0]));
    CHECK(47.43 == approx(det.get()[1]));
    CHECK(57.14 == approx(det.get()[2]));
    CHECK(-13.37 == approx(det.get()[3]));
  }
}
