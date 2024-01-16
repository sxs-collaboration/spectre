// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/CombineSpacetimeView.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"

namespace {

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.CombineSpacetimeView",
                  "[DataStructures][Unit]") {
  const size_t SpatialDim = 3;
  const DataVector used_for_size(5);
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);
  const auto nn_dist = make_not_null(&distribution);

  tnsr::a<double, SpatialDim, Frame::Inertial> test_spacetime_vector;
  const auto scalar_time_component =
      make_with_random_values<Scalar<double>>(nn_gen, nn_dist, used_for_size);
  const auto vector_spatial_components =
      make_with_random_values<tnsr::i<double, SpatialDim, Frame::Inertial>>(
          nn_gen, nn_dist, used_for_size);

  combine_spacetime_view<SpatialDim, UpLo::Lo, Frame::Inertial>(
      make_not_null(&test_spacetime_vector), scalar_time_component,
      vector_spatial_components);

  CHECK(test_spacetime_vector.get(0) == get(scalar_time_component));
  for (size_t i = 0; i < SpatialDim; ++i) {
    CHECK(test_spacetime_vector.get(i + 1) == vector_spatial_components.get(i));
  }

  tnsr::Abb<DataVector, SpatialDim, Frame::Inertial> test_spacetime_tensor;
  const auto tensor_time_component = make_with_random_values<
      tnsr::aa<DataVector, SpatialDim, Frame::Inertial>>(nn_gen, nn_dist,
                                                         used_for_size);
  const auto tensor_spatial_components = make_with_random_values<
      tnsr::Iaa<DataVector, SpatialDim, Frame::Inertial>>(nn_gen, nn_dist,
                                                          used_for_size);

  combine_spacetime_view<SpatialDim, UpLo::Up, Frame::Inertial>(
      make_not_null(&test_spacetime_tensor), tensor_time_component,
      tensor_spatial_components);

  for (size_t i = 0; i <= SpatialDim; ++i) {
    for (size_t j = 0; j <= SpatialDim; ++j) {
      CHECK(test_spacetime_tensor.get(0, i, j) ==
            tensor_time_component.get(i, j));
      for (size_t k = 0; k < SpatialDim; ++k) {
        CHECK(test_spacetime_tensor.get(k + 1, i, j) ==
              tensor_spatial_components.get(k, i, j));
      }
    }
  }
}
}  // namespace
