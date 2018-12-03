// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

template <typename TagsList>
class Variables;

namespace {

template <size_t Dim>
struct Var1 {
  using type = tnsr::i<DataVector, Dim, Frame::Grid>;
};

struct Var2 {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
using two_vars = tmpl::list<Var1<Dim>, Var2>;

template <size_t Dim>
using one_var = tmpl::list<Var1<Dim>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.Transpose",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  // clang-format off
  /// [transpose_matrix]
  const DataVector matrix{ 1.,  2.,  3.,
                           4.,  5.,  6.,
                           7.,  8.,  9.,
                          10., 11., 12.};
  CHECK(transpose(matrix, 3, 4) == DataVector{1.,  4.,  7., 10.,
                                              2.,  5.,  8., 11.,
                                              3.,  6.,  9., 12.});
  /// [transpose_matrix]
  // clang-format on

  DataVector transposed_matrix(12);
  raw_transpose(make_not_null(transposed_matrix.data()), matrix.data(), 3, 4);
  CHECK(transposed_matrix ==
        DataVector{1., 4., 7., 10., 2., 5., 8., 11., 3., 6., 9., 12.});

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
  std::fill(transposed_data.begin(), transposed_data.end(), 0.0);
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
  const size_t chunk_size_vars = 8;
  const size_t n_grid_pts = 2 * chunk_size_vars;
  Variables<two_vars<2>> variables(n_grid_pts, 0.);
  for (size_t i = 0; i < variables.size(); ++i) {
    // clang-tidy: pointer arithmetic
    variables.data()[i] = i * i;  // NOLINT
  }
  const size_t number_of_chunks_vars = variables.size() / chunk_size_vars;
  auto transposed_vars = variables;
  transpose(make_not_null(&transposed_vars), variables, chunk_size_vars,
            number_of_chunks_vars);
  for (size_t i = 0; i < chunk_size_vars; ++i) {
    for (size_t j = 0; j < number_of_chunks_vars; ++j) {
      // clang-tidy: pointer arithmetic
      CHECK(variables.data()[i + chunk_size_vars * j] ==             // NOLINT
            transposed_vars.data()[j + number_of_chunks_vars * i]);  // NOLINT
    }
  }
  /// [transpose_by_not_null_example]

  /// [partial_transpose_example]
  Variables<one_var<2>> partial_vars(n_grid_pts, 0.);
  get<Var1<2>>(partial_vars) = get<Var1<2>>(variables);
  Variables<one_var<2>> partial_transpose(n_grid_pts, 0.);
  const size_t partial_number_of_chunks = 2*number_of_chunks_vars / 3;
  transpose(make_not_null(&partial_transpose), variables, chunk_size_vars,
            partial_number_of_chunks);
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

  const auto another_partial_transpose =
      transpose<Variables<two_vars<2>>, Variables<one_var<2>>>(
          variables, chunk_size_vars, partial_number_of_chunks);
  for (size_t i = 0; i < chunk_size_vars; ++i) {
    for (size_t j = 0; j < partial_number_of_chunks; ++j) {
      // clang-tidy: pointer arithmetic
      CHECK(another_partial_transpose
                .data()[j + partial_number_of_chunks * i] ==  // NOLINT
            variables.data()[i + chunk_size_vars * j]);       // NOLINT
      CHECK(another_partial_transpose
                .data()[j + partial_number_of_chunks * i] ==  // NOLINT
            partial_vars.data()[i + chunk_size_vars * j]);    // NOLINT
    }
  }
}
