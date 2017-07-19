// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/EagerMath/DivideBy.hpp"

TEST_CASE("Unit.DataStructures.Tensor.DivideBy", "[Functors][Unit]") {
  const size_t npts = 2;
  const DataVector one(npts, 1.0);
  const DataVector two(npts, 2.0);
  const DataVector three(npts, 3.0);
  const DataVector four(npts, 4.0);
  const DataVector five(npts, 5.0);
  const DataVector twelve(npts, 12.0);
  const DataVector thirteen(npts, 13.0);

  const Tensor<DataVector, Symmetry<1>,
               typelist<SpatialIndex<1, UpLo::Lo, Frame::Grid>>>
      one_d_covector{{{two}}};
  const DataVector magnitude_one_d_covector = two;
  const auto normalized_one_d_covector =
      divide_by(one_d_covector, magnitude_one_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(normalized_one_d_covector.get<0>()[s] == 1.0);
  }

  const Tensor<DataVector, Symmetry<1>,
               typelist<SpacetimeIndex<1, UpLo::Up, Frame::Grid>>>
      two_d_vector{{{three, four}}};
  const DataVector magnitude_two_d_vector = five;
  const auto normalized_two_d_vector =
      divide_by(two_d_vector, magnitude_two_d_vector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(normalized_two_d_vector.get<0>()[s] == 0.6);
    CHECK(normalized_two_d_vector.get<1>()[s] == 0.8);
  }

  const Tensor<DataVector, Symmetry<1>,
               typelist<SpatialIndex<2, UpLo::Up, Frame::Grid>>>
      two_d_spatial_vector{{{five, twelve}}};
  const DataVector magnitude_two_d_spatial_vector = thirteen;
  const auto normalized_two_d_spatial_vector =
      divide_by(two_d_spatial_vector, magnitude_two_d_spatial_vector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(normalized_two_d_spatial_vector.get<0>()[s] == 5.0 / 13.0);
    CHECK(normalized_two_d_spatial_vector.get<1>()[s] == 12.0 / 13.0);
  }

  const Tensor<DataVector, Symmetry<1>,
               typelist<SpatialIndex<3, UpLo::Lo, Frame::Grid>>>
      three_d_covector{{{three, twelve, four}}};
  const DataVector magnitude_three_d_covector = thirteen;
  const auto normalized_three_d_covector =
      divide_by(three_d_covector, magnitude_three_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(normalized_three_d_covector.get<0>()[s] == 3.0 / 13.0);
    CHECK(normalized_three_d_covector.get<1>()[s] == 12.0 / 13.0);
    CHECK(normalized_three_d_covector.get<2>()[s] == 4.0 / 13.0);
  }

  const Tensor<DataVector, Symmetry<1>,
               typelist<SpacetimeIndex<4, UpLo::Lo, Frame::Grid>>>
      five_d_covector{{{two, twelve, four, one, two}}};
  const DataVector magnitude_five_d_covector = thirteen;
  const auto normalized_five_d_covector =
      divide_by(five_d_covector, magnitude_five_d_covector);
  for (size_t s = 0; s < npts; ++s) {
    CHECK(normalized_five_d_covector.get<0>()[s] == 2.0 / 13.0);
    CHECK(normalized_five_d_covector.get<1>()[s] == 12.0 / 13.0);
    CHECK(normalized_five_d_covector.get<2>()[s] == 4.0 / 13.0);
    CHECK(normalized_five_d_covector.get<3>()[s] == 1.0 / 13.0);
    CHECK(normalized_five_d_covector.get<4>()[s] == 2.0 / 13.0);
  }
}
