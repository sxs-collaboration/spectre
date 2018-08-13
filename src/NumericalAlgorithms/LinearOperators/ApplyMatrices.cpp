// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/ApplyMatrices.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace {
void multiply_in_first_dimension(const gsl::not_null<double*> result,
                                 const gsl::not_null<size_t*> data_size,
                                 const Matrix& matrix,
                                 const double* data) noexcept {
  *data_size /= matrix.columns();
  dgemm_<true>('N', 'N',
               matrix.rows(),     // rows of matrix and result
               *data_size,        // columns of result and u
               matrix.columns(),  // columns of matrix and rows of u
               1.0,               // overall multiplier
               matrix.data(),     // matrix
               matrix.rows(),     // rows of matrix
               data,              // u
               matrix.columns(),  // rows of u
               0.0,               // multiplier for unused term
               result.get(),      // result
               matrix.rows());    // rows of result
  *data_size *= matrix.rows();
}

void do_transpose(const gsl::not_null<double*> result, const double* const data,
                  const size_t data_size, const size_t chunk_size) noexcept {
  raw_transpose(result, data, chunk_size, data_size / chunk_size);
}

struct Scratch {
  std::vector<double> buffer;
  double* a;
  double* b;
};

// This does not take into account the order that the matrices are
// applied in and gives the largest amount of space that could be
// required for any application order.
template <typename MatrixType, size_t Dim>
Scratch get_scratch(const std::array<MatrixType, Dim>& matrices,
                    const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
  size_t size = number_of_independent_components;
  for (size_t i = 0; i < Dim; ++i) {
    const auto& matrix = gsl::at(matrices, i);
    if (dereference_wrapper(matrix) == Matrix{}) {
      size *= extents[i];
    } else {
      size *= std::max(dereference_wrapper(matrix).rows(),
                       dereference_wrapper(matrix).columns());
    }
  }
  Scratch result{};
  result.buffer.resize(2 * size);
  result.a = &result.buffer[0];
  result.b = &result.buffer[size];
  return result;
}

// Produce the array of the number of rows in each matrix.  Empty
// matrices are processed as if they were identity matrices.
template <size_t Dim, typename MatrixType>
std::array<size_t, Dim> matrix_rows(const std::array<MatrixType, Dim>& matrices,
                                    const Index<Dim>& extents) noexcept {
  std::array<size_t, Dim> result{};
  for (size_t i = 0; i < Dim; ++i) {
    if (dereference_wrapper(gsl::at(matrices, i)) == Matrix{}) {
      gsl::at(result, i) = extents[i];
    } else {
      gsl::at(result, i) = dereference_wrapper(gsl::at(matrices, i)).rows();
    }
  }
  return result;
}
}  // namespace

namespace apply_matrices_detail {
template <size_t Dim, bool... DimensionIsIdentity>
template <typename MatrixType>
void Impl<Dim, DimensionIsIdentity...>::apply(
    const gsl::not_null<double*> result,
    const std::array<MatrixType, Dim>& matrices, const double* const data,
    const Index<Dim>& extents,
    const size_t number_of_independent_components) noexcept {
  if (dereference_wrapper(matrices[sizeof...(DimensionIsIdentity)]) ==
      Matrix{}) {
    Impl<Dim, DimensionIsIdentity..., true>::apply(
        result, matrices, data, extents, number_of_independent_components);
  } else {
    Impl<Dim, DimensionIsIdentity..., false>::apply(
        result, matrices, data, extents, number_of_independent_components);
  }
}

template <>
struct Impl<0> {
  static constexpr const size_t Dim = 0;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& /*matrices*/,
                    const double* const data, const Index<Dim>& /*extents*/,
                    const size_t number_of_independent_components) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::copy(data, data + number_of_independent_components, result.get());
  }
};

template <>
struct Impl<1, false> {
  static constexpr const size_t Dim = 1;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(result, &data_size, matrices[0], data);
  }
};

template <>
struct Impl<1, true> {
  static constexpr const size_t Dim = 1;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& /*matrices*/,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::copy(data, data + number_of_independent_components * extents.product(),
              result.get());
  }
};

template <>
struct Impl<2, false, false> {
  static constexpr size_t Dim = 2;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(scratch.a, &data_size, matrices[0], data);
    do_transpose(scratch.b, scratch.a, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[1]);
  }
};

template <>
struct Impl<2, false, true> {
  static constexpr size_t Dim = 2;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(result, &data_size, matrices[0], data);
  }
};

template <>
struct Impl<2, true, false> {
  static constexpr size_t Dim = 2;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    do_transpose(scratch.b, data, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[1]);
  }
};

template <>
struct Impl<2, true, true> {
  static constexpr size_t Dim = 2;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& /*matrices*/,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::copy(data, data + number_of_independent_components * extents.product(),
              result.get());
  }
};

template <>
struct Impl<3, false, false, false> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(scratch.a, &data_size, matrices[0], data);
    do_transpose(scratch.b, scratch.a, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(scratch.b, scratch.a, data_size, rows[1]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[2], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[2]);
  }
};

template <>
struct Impl<3, false, false, true> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(scratch.a, &data_size, matrices[0], data);
    do_transpose(scratch.b, scratch.a, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[1] * rows[2]);
  }
};

template <>
struct Impl<3, false, true, false> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(scratch.a, &data_size, matrices[0], data);
    do_transpose(scratch.b, scratch.a, data_size, rows[0] * rows[1]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[2], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[2]);
  }
};

template <>
struct Impl<3, false, true, true> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    size_t data_size = number_of_independent_components * extents.product();
    multiply_in_first_dimension(result, &data_size, matrices[0], data);
  }
};

template <>
struct Impl<3, true, false, false> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    do_transpose(scratch.b, data, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(scratch.b, scratch.a, data_size, rows[1]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[2], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[2]);
  }
};

template <>
struct Impl<3, true, false, true> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    do_transpose(scratch.b, data, data_size, rows[0]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[1], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[1] * rows[2]);
  }
};

template <>
struct Impl<3, true, true, false> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& matrices,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    const auto rows = matrix_rows(matrices, extents);
    auto scratch =
        get_scratch(matrices, extents, number_of_independent_components);

    size_t data_size = number_of_independent_components * extents.product();
    do_transpose(scratch.b, data, data_size, rows[0] * rows[1]);
    multiply_in_first_dimension(scratch.a, &data_size, matrices[2], scratch.b);
    do_transpose(result, scratch.a, data_size,
                 number_of_independent_components * rows[2]);
  }
};

template <>
struct Impl<3, true, true, true> {
  static constexpr size_t Dim = 3;
  template <typename MatrixType>
  static void apply(const gsl::not_null<double*> result,
                    const std::array<MatrixType, Dim>& /*matrices*/,
                    const double* const data, const Index<Dim>& extents,
                    const size_t number_of_independent_components) noexcept {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::copy(data, data + number_of_independent_components * extents.product(),
              result.get());
  }
};

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define MATRIX(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                           \
  template void Impl<DIM(data)>::apply(                                \
      const gsl::not_null<double*>,                                    \
      const std::array<MATRIX(data), DIM(data)>&, const double* const, \
      const Index<DIM(data)>&, const size_t) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (0, 1, 2, 3),
                        (Matrix, std::reference_wrapper<const Matrix>))

#undef DIM
#undef MATRIX
#undef INSTANTIATE
}  // namespace apply_matrices_detail
