// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/DataBoxTag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Numerical/LinearOperators/Transpose.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct Data1 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "Data1";
};
struct Data2 : db::DataBoxTag {
  using type = Scalar<DataVector>;
  static constexpr db::DataBoxString_t label = "Data2";
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Transpose",
                  "[Numerical][LinearOperators][Unit]") {
  /// [return_transpose_example]
  const size_t chunk_size = 8;
  const size_t number_of_chunks = 2;
  const size_t n_pts = chunk_size * number_of_chunks;
  DataVector data(n_pts);
  for (size_t i = 0; i < data.size(); ++i) {
    data[i] = i * i;
  }
  DataVector transposed_data(n_pts, 0.);
  transposed_data = transpose(data, chunk_size, number_of_chunks);
  for (size_t i = 0; i < chunk_size; ++i) {
    for (size_t j = 0; j < number_of_chunks; ++j) {
      CHECK(data[i + chunk_size * j] ==
            transposed_data[j + number_of_chunks * i]);
    }
  }
  /// [return_transpose_example]
  std::fill(transposed_data.begin(),transposed_data.end(),0.0);
  DataVector ref_to_data;
  ref_to_data.set_data_ref(&data);
  DataVector ref_to_transposed_data;
  ref_to_transposed_data.set_data_ref(&transposed_data);
  CHECK(not ref_to_data.is_owning());
  CHECK(not ref_to_transposed_data.is_owning());
  ref_to_transposed_data = transpose(ref_to_data, chunk_size, number_of_chunks);
  for (size_t i = 0; i < chunk_size; ++i) {
    for (size_t j = 0; j < number_of_chunks; ++j) {
      // clang-tidy: pointer arithmetic
      CHECK(ref_to_data[i + chunk_size * j] ==                  // NOLINT
            ref_to_transposed_data[j + number_of_chunks * i]);  // NOLINT
      CHECK(data[i + chunk_size * j] ==                         // NOLINT
            ref_to_transposed_data[j + number_of_chunks * i]);  // NOLINT
    }
  }

  /// [transpose_by_not_null_example]
  const size_t n_grid_pts = 16;
  Variables<typelist<Data1, Data2>> variables(n_grid_pts, 0.);
  for (size_t i = 0; i < variables.size(); ++i) {
    // clang-tidy: pointer arithmetic
    variables.data()[i] = i * i;  // NOLINT
  }
  const size_t chunk_size_vars = 8;
  const size_t number_of_chunks_vars = 4;
  auto transposed_vars = variables;
  transpose(variables, chunk_size_vars, number_of_chunks_vars,
            make_not_null(&transposed_vars));
  for (size_t i = 0; i < chunk_size_vars; ++i) {
    for (size_t j = 0; j < number_of_chunks_vars; ++j) {
      // clang-tidy: pointer arithmetic
      CHECK(variables.data()[i + chunk_size_vars * j] ==             // NOLINT
            transposed_vars.data()[j + number_of_chunks_vars * i]);  // NOLINT
    }
  }
  /// [transpose_by_not_null_example]

  /// [partial_transpose_example]
  Variables<typelist<Data1>> partial_vars(n_grid_pts, 0.);
  partial_vars.template get<Data1>() = variables.template get<Data1>();
  Variables<typelist<Data1>> partial_transpose(n_grid_pts, 0.);
  const size_t partial_number_of_chunks = number_of_chunks_vars / 2;
  transpose(variables, chunk_size_vars, partial_number_of_chunks,
            make_not_null(&partial_transpose));
  for (size_t i = 0; i < chunk_size_vars; ++i) {
    for (size_t j = 0; j < partial_number_of_chunks; ++j) {
      // clang-tidy: pointer arithmetic
      CHECK(partial_transpose
                .data()[j + partial_number_of_chunks * i] ==  // NOLINT
            variables.data()[i + chunk_size_vars * j]);       // NOLINT
      CHECK(partial_transpose
                .data()[j + partial_number_of_chunks * i] ==  // NOLINT
            partial_vars.data()[i + chunk_size_vars * j]);    // NOLINT
    }
  }
  /// [partial_transpose_example]
}
