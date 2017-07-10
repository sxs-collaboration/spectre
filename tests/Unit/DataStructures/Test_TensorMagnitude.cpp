// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"

TEST_CASE("Unit.DataStructures.Tensor.EuclideanMagnitude",
          "[DataStructures][Unit]") {
  const size_t npts = 8;
  DataVector one(npts, 1.0);
  DataVector two(npts, 2.0);
  DataVector minus_three(npts, -3.0);
  DataVector four(npts, 4.0);
  DataVector minus_five(npts, -5.0);
  DataVector twelve(npts, 12.0);
  DataVector thirteen(npts, 13.0);

  tnsr::i<DataVector, 1, Frame::Grid> one_d_covector;
  one_d_covector.get<0>() = two;
  DataVector magnitude_one_d_covector = magnitude(one_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_one_d_covector[s] == 2.0);
  }

  tnsr::A<DataVector, 1, Frame::Grid> one_d_vector;
  one_d_vector.get<0>() = minus_three;
  one_d_vector.get<1>() = four;
  DataVector magnitude_one_d_vector = magnitude(one_d_vector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_one_d_vector[s] == 5.0);
  }

  tnsr::I<DataVector, 2, Frame::Grid> two_d_vector;
  two_d_vector.get<0>() = minus_five;
  two_d_vector.get<1>() = twelve;
  DataVector magnitude_two_d_vector = magnitude(two_d_vector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_two_d_vector[s] == 13.0);
  }

  tnsr::i<DataVector, 3, Frame::Grid> three_d_covector;
  three_d_covector.get<0>() = minus_three;
  three_d_covector.get<1>() = twelve;
  three_d_covector.get<2>() = four;
  DataVector magnitude_three_d_covector = magnitude(three_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_three_d_covector[s] == 13.0);
  }

  // 5D example
  tnsr::a<DataVector, 4, Frame::Grid> five_d_covector;
  five_d_covector.get<0>() = two;
  five_d_covector.get<1>() = twelve;
  five_d_covector.get<2>() = four;
  five_d_covector.get<3>() = one;
  five_d_covector.get<4>() = two;
  DataVector magnitude_five_d_covector = magnitude(five_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_five_d_covector[s] == 13.0);
  }
}

TEST_CASE("Unit.DataStructures.Tensor.Magnitude", "[DataStructures][Unit]") {
  const size_t npts = 8;
  DataVector one(npts, 1.0);
  DataVector two(npts, 2.0);
  DataVector minus_three(npts, -3.0);
  DataVector four(npts, 4.0);
  DataVector minus_five(npts, -5.0);
  DataVector twelve(npts, 12.0);
  DataVector thirteen(npts, 13.0);

  tnsr::i<DataVector, 1, Frame::Grid> one_d_covector;
  tnsr::II<DataVector, 1, Frame::Grid> inv_h;
  one_d_covector.get<0>() = two;
  inv_h.get<0, 0>() = four;
  DataVector magnitude_a = magnitude(one_d_covector, inv_h);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_a[s] == 4.0);
  }
  tnsr::i<DataVector, 3, Frame::Grid> three_d_covector;
  tnsr::II<DataVector, 3, Frame::Grid> inv_g;
  three_d_covector.get<0>() = minus_three;
  three_d_covector.get<1>() = twelve;
  three_d_covector.get<2>() = four;
  inv_g.get<0, 0>() = two;
  inv_g.get<0, 1>() = minus_three;
  inv_g.get<0, 2>() = four;
  inv_g.get<1, 1>() = minus_five;
  inv_g.get<1, 2>() = twelve;
  inv_g.get<2, 2>() = thirteen;
  DataVector magnitude_three_d_covector = magnitude(three_d_covector, inv_g);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(magnitude_three_d_covector[s] == sqrt(778.0));
  }
}
