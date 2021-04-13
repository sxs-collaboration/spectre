// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"

namespace {

template <typename T>
void check_results(const DynamicBuffer<T>& dynamic_buffer,
                   const std::vector<T>& expected) {
  CHECK(dynamic_buffer.size() == expected.size());

  for (size_t i = 0; i < dynamic_buffer.size(); ++i) {
    CHECK(dynamic_buffer[i] == expected[i]);
    CHECK(dynamic_buffer.at(i) == expected[i]);
  }

  size_t i = 0;
  for (auto it = dynamic_buffer.begin(); it != dynamic_buffer.end();
       ++it, ++i) {
    CHECK(*it == expected[i]);
  }

  i = 0;
  for (const auto& val : dynamic_buffer) {
    CHECK(val == expected[i]);
    ++i;
  }
}

template <typename T>
std::vector<T> create_random_data(const gsl::not_null<std::mt19937*> gen,
                                  const size_t number_of_vectors,
                                  const size_t number_of_grid_points) {
  auto value_distribution = std::uniform_real_distribution(
      std::numeric_limits<double>::min(), std::numeric_limits<double>::max());

  std::vector<T> res(number_of_vectors);
  for (size_t i = 0; i < number_of_vectors; ++i) {
    res[i] = make_with_random_values<T>(gen, value_distribution,
                                        number_of_grid_points);
  }
  return res;
}

template <typename T>
void test_dynamic_buffer() {
  MAKE_GENERATOR(gen);
  auto size_distribution = std::uniform_int_distribution(1, 10);
  auto number_of_grid_points = static_cast<size_t>(size_distribution(gen));
  const auto number_of_vectors = static_cast<size_t>(size_distribution(gen));
  if constexpr (std::is_same_v<double, T>) {
    number_of_grid_points = 1;
  }

  CAPTURE(number_of_grid_points);
  CAPTURE(number_of_vectors);

  const auto expected = create_random_data<T>(
      make_not_null(&gen), number_of_vectors, number_of_grid_points);

  DynamicBuffer<T> dynamic_buffer(number_of_vectors, number_of_grid_points);

  for (size_t i = 0; i < number_of_vectors; ++i) {
    dynamic_buffer.at(i) = expected.at(i);
  }

  dynamic_buffer = serialize_and_deserialize(dynamic_buffer);
  check_results(dynamic_buffer, expected);

  test_copy_semantics(dynamic_buffer);
  auto dynamic_buffer_for_move = dynamic_buffer;
  test_move_semantics(std::move(dynamic_buffer_for_move), dynamic_buffer);

  DynamicBuffer<T> dynamic_buffer_copied(dynamic_buffer);
  DynamicBuffer<T> dynamic_buffer_assigned;
  dynamic_buffer_assigned = dynamic_buffer;

  if constexpr (tt::is_iterable_v<T>) {
    auto copied_it = dynamic_buffer_copied.begin();
    auto assigned_it = dynamic_buffer_assigned.begin();
    // check that the new buffers don't point to the same data storage location
    for (auto it = dynamic_buffer.begin(); it != dynamic_buffer.end();
         ++it, ++copied_it, ++assigned_it) {
      // first `DataVector` of dynamic_buffer
      const auto first_data_vector = *it;

      // underlying C array
      const auto array = first_data_vector.data();

      // address of first element
      const auto address = &*array;

      // analogously for copied DynamicBuffers
      CHECK(&*((*copied_it).data()) != address);
      CHECK(&*((*assigned_it).data()) != address);
    }
    CHECK(copied_it == dynamic_buffer_copied.end());
    CHECK(assigned_it == dynamic_buffer_assigned.end());
  }

  DynamicBuffer<T> dynamic_buffer_moved = std::move(dynamic_buffer);

  dynamic_buffer_copied = serialize_and_deserialize(dynamic_buffer_copied);
  dynamic_buffer_assigned = serialize_and_deserialize(dynamic_buffer_assigned);
  dynamic_buffer_moved = serialize_and_deserialize(dynamic_buffer_moved);

  check_results(dynamic_buffer_copied, expected);
  check_results(dynamic_buffer_assigned, expected);
  check_results(dynamic_buffer_moved, expected);
}

}  // namespace
SPECTRE_TEST_CASE("Unit.DataStructures.DynamicBuffer",
                  "[DataStructures][Unit]") {
  for (size_t i = 0; i < 20; ++i) {
    test_dynamic_buffer<DataVector>();
    test_dynamic_buffer<double>();
  }
}
