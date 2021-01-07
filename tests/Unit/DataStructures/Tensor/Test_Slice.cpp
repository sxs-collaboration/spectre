// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace {
template <typename VectorType>
void test_variables_slice() noexcept {
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 10};

  const size_t x_extents = sdist(gen);
  const size_t y_extents = sdist(gen);
  const size_t z_extents = sdist(gen);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>> vars{
      x_extents * y_extents * z_extents};
  const size_t tensor_size =
      TestHelpers::Tags::Vector<VectorType>::type::size();
  Index<3> extents(x_extents, y_extents, z_extents);

  // Test data_on_slice function by using a predictable data set where each
  // entry is assigned a value equal to its index
  for (size_t s = 0; s < vars.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    vars.data()[s] = s;  // NOLINT
  }
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_x(y_extents * z_extents, 0.);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_y(x_extents * z_extents, 0.);
  Variables<tmpl::list<TestHelpers::Tags::Vector<VectorType>>>
      expected_vars_sliced_in_z(x_extents * y_extents, 0.);
  const size_t x_offset = sdist(gen) % x_extents;
  const size_t y_offset = sdist(gen) % y_extents;
  const size_t z_offset = sdist(gen) % z_extents;

  for (size_t s = 0; s < expected_vars_sliced_in_x.size(); ++s) {
    // clang-tidy: do not use pointer arithmetic
    expected_vars_sliced_in_x.data()[s] = x_offset + s * x_extents;  // NOLINT
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t z = 0; z < z_extents; ++z) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_y
            .data()[x + x_extents * (z + z_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y_offset + z * y_extents);
      }
    }
  }
  for (size_t i = 0; i < tensor_size; ++i) {
    for (size_t x = 0; x < x_extents; ++x) {
      for (size_t y = 0; y < y_extents; ++y) {
        // clang-tidy: do not use pointer arithmetic
        expected_vars_sliced_in_z
            .data()[x + x_extents * (y + y_extents * i)] =  // NOLINT
            i * extents.product() + x + x_extents * (y + y_extents * z_offset);
      }
    }
  }

  INFO("Test simple slice");
  CHECK(data_on_slice(get<TestHelpers::Tags::Vector<VectorType>>(vars), extents,
                      0, x_offset) ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_x));
  CHECK(data_on_slice(get<TestHelpers::Tags::Vector<VectorType>>(vars), extents,
                      1, y_offset) ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_y));
  CHECK(data_on_slice(get<TestHelpers::Tags::Vector<VectorType>>(vars), extents,
                      2, z_offset) ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_z));

  INFO("Test slice of a std::optional<Tensor>");
  REQUIRE(data_on_slice(std::make_optional(
                            get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                        extents, 0, x_offset)
              .has_value());
  CHECK(data_on_slice(std::make_optional(
                          get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                      extents, 0, x_offset)
            .value() ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_x));
  REQUIRE(data_on_slice(std::make_optional(
                            get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                        extents, 1, y_offset)
              .has_value());
  CHECK(data_on_slice(std::make_optional(
                          get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                      extents, 1, y_offset)
            .value() ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_y));
  REQUIRE(data_on_slice(std::make_optional(
                            get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                        extents, 2, z_offset)
              .has_value());
  CHECK(data_on_slice(std::make_optional(
                          get<TestHelpers::Tags::Vector<VectorType>>(vars)),
                      extents, 2, z_offset)
            .value() ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_z));

  CHECK_FALSE(
      data_on_slice(std::optional<tnsr::I<VectorType, 3, Frame::Inertial>>{},
                    extents, 0, x_offset)
          .has_value());

  // Test not_null<boost::option<Tensor>>
  using TensorType = tnsr::I<VectorType, 3>;
  auto optional_tensor = std::make_optional(TensorType{});
  CHECK(optional_tensor.has_value());
  data_on_slice(make_not_null(&optional_tensor), std::optional<TensorType>{},
                extents, 0, x_offset);
  CHECK_FALSE(optional_tensor.has_value());

  data_on_slice(
      make_not_null(&optional_tensor),
      std::make_optional(get<TestHelpers::Tags::Vector<VectorType>>(vars)),
      extents, 0, x_offset);
  REQUIRE(optional_tensor.has_value());
  CHECK(optional_tensor.value() ==
        get<TestHelpers::Tags::Vector<VectorType>>(expected_vars_sliced_in_x));

  optional_tensor = std::nullopt;
  CHECK_FALSE(optional_tensor.has_value());
  data_on_slice(make_not_null(&optional_tensor), std::optional<TensorType>{},
                extents, 0, x_offset);
  CHECK_FALSE(optional_tensor.has_value());
}

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Slice",
                  "[DataStructures][Unit]") {
  test_variables_slice<ComplexDataVector>();
  test_variables_slice<ComplexModalVector>();
  test_variables_slice<DataVector>();
  test_variables_slice<ModalVector>();
}
}  // namespace
