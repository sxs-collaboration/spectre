// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include <array>
#include <blaze/math/DynamicMatrix.h>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace partial_derivatives_detail {
void apply_matrix_in_first_dim(double* result, const double* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  dgemm_<true>(
      'N', 'N',
      matrix.rows(),              // rows of matrix and result
      size / matrix.columns(),    // columns of result and input
      matrix.columns(),           // columns of matrix and rows of input
      1.0,                        // overall multiplier
      matrix.data(),              // matrix
      matrix.spacing(),           // rows of matrix including padding
      input,                      // input
      matrix.columns(),           // rows of input
      add_to_result ? 1.0 : 0.0,  // overwrite output with result or add to it
      result,                     // result
      matrix.rows());             // rows of result
}
void apply_matrix_in_first_dim(std::complex<double>* result,
                               const std::complex<double>* const input,
                               const Matrix& matrix, const size_t size,
                               const bool add_to_result) {
  // BLAS zgemm operates on complex matrices, so we need to copy the real matrix
  // to a complex matrix with zero imaginary part before calling zgemm.
  // Possible performance optimization: avoid the copy here by storing the
  // complex matrix in a static cache. We probably only want to add this to
  // Spectral.hpp once profiling shows that it becomes necessary.
  const blaze::DynamicMatrix<std::complex<double>, blaze::columnMajor>
      matrix_complex{matrix};
  zgemm_<true>('N', 'N',
               matrix.rows(),            // rows of matrix and result
               size / matrix.columns(),  // columns of result and input
               matrix.columns(),         // columns of matrix and rows of input
               std::complex{1.0, 0.0},   // overall multiplier
               matrix_complex.data(),    // matrix
               matrix.spacing(),         // rows of matrix including padding
               input,                    // input
               matrix.columns(),         // rows of input
               std::complex{add_to_result ? 1.0 : 0.0,
                            0.0},  // overwrite output with result or add to it
               result,             // result
               matrix.rows());     // rows of result
  // This implementation is ~1.35x slower than the implementation above (based
  // on the "Partial derivatives complex" benchmark in
  // Test_PartialDerivatives.cpp).
  //   DataVector buffer(size * 2);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 reinterpret_cast<const double*>(input), 2, size);
  //   apply_matrix_in_first_dim(buffer.data(),
  //                             reinterpret_cast<const double*>(result),
  //                             matrix, size * 2, add_to_result);
  //   raw_transpose(make_not_null(reinterpret_cast<double*>(result)),
  //                 buffer.data(), size, 2);
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
namespace {
// Fixed-size differentiation kernels. Each kernel is templated on the extent
// of the dimension it contracts over (needed at compile time so the
// accumulators live in vector registers); every other extent is a runtime
// batch count over unit-stride data. Written as plain FMA loops that
// auto-vectorize (AVX-512/AVX2/NEON) with no BLAS packing and no transposes.

// Largest extent handled by the fast path; larger meshes fall back to the
// generic dgemm+transpose implementation.
constexpr size_t fast_path_max_extent = 20;

// Calls f(std::integral_constant<size_t, extent>{}) for extents in the
// instantiated kernel range. Returns false outside the range so callers can
// fall back to the generic implementation.
template <typename F>
bool dispatch_on_extent(const size_t extent, const F& f) {
  switch (extent) {
    case 2:
      f(std::integral_constant<size_t, 2>{});
      return true;
    case 3:
      f(std::integral_constant<size_t, 3>{});
      return true;
    case 4:
      f(std::integral_constant<size_t, 4>{});
      return true;
    case 5:
      f(std::integral_constant<size_t, 5>{});
      return true;
    case 6:
      f(std::integral_constant<size_t, 6>{});
      return true;
    case 7:
      f(std::integral_constant<size_t, 7>{});
      return true;
    case 8:
      f(std::integral_constant<size_t, 8>{});
      return true;
    case 9:
      f(std::integral_constant<size_t, 9>{});
      return true;
    case 10:
      f(std::integral_constant<size_t, 10>{});
      return true;
    case 11:
      f(std::integral_constant<size_t, 11>{});
      return true;
    case 12:
      f(std::integral_constant<size_t, 12>{});
      return true;
    case 13:
      f(std::integral_constant<size_t, 13>{});
      return true;
    case 14:
      f(std::integral_constant<size_t, 14>{});
      return true;
    case 15:
      f(std::integral_constant<size_t, 15>{});
      return true;
    case 16:
      f(std::integral_constant<size_t, 16>{});
      return true;
    case 17:
      f(std::integral_constant<size_t, 17>{});
      return true;
    case 18:
      f(std::integral_constant<size_t, 18>{});
      return true;
    case 19:
      f(std::integral_constant<size_t, 19>{});
      return true;
    case 20:
      f(std::integral_constant<size_t, 20>{});
      return true;
    default:
      return false;
  }
}

// The kernels index the differentiation matrix with a column stride equal to
// their compile-time extent, so the (tiny) matrix is copied out of blaze
// storage (which may pad columns) once per call. A runtime column stride
// measurably slows the kernels down with gcc.
using PackedMatrix =
    std::array<double, fast_path_max_extent * fast_path_max_extent>;

PackedMatrix pack_matrix(const Matrix& matrix) {
  PackedMatrix packed{};
  const size_t extent = matrix.rows();
  for (size_t k = 0; k < extent; ++k) {
    for (size_t i = 0; i < extent; ++i) {
      packed[i + (extent * k)] = matrix(i, k);
    }
  }
  return packed;
}

// Differentiate along the first (fastest) logical dimension:
// out(i, l) = sum_k D(i, k) u(k, l) for `number_of_lines` contiguous lines of
// length N. The output line is accumulated in registers, one broadcast-FMA of
// a matrix column per k.
template <size_t N>
void differentiate_first_dim(double* const out, const double* const u,
                             const double* const matrix,
                             const size_t number_of_lines) {
  for (size_t line = 0; line < number_of_lines; ++line) {
    const double* const u_line = u + (line * N);
    double* const out_line = out + (line * N);
    std::array<double, N> accumulator{};
    for (size_t k = 0; k < N; ++k) {
      const double u_k = u_line[k];
      const double* const matrix_column = matrix + (k * N);
      for (size_t i = 0; i < N; ++i) {
        accumulator[i] += matrix_column[i] * u_k;
      }
    }
    for (size_t i = 0; i < N; ++i) {
      out_line[i] = accumulator[i];
    }
  }
}

// Differentiate along a middle or the slowest logical dimension with the
// stride (the product of the extents of all faster dimensions) known at
// compile time. The data is viewed as [batch][k][Stride] with the contracted
// index k in the middle: out(s, i, b) = sum_k D(i, k) u(s, k, b). Each output
// line of `Stride` values is accumulated in registers and stored exactly once
// (a read-modify-write pass per k measured ~1.6x slower).
template <size_t N, size_t Stride>
void differentiate_middle_dim_small_stride(double* const out,
                                           const double* const u,
                                           const double* const matrix,
                                           const size_t number_of_batches) {
  for (size_t b = 0; b < number_of_batches; ++b) {
    const double* const u_batch = u + (b * Stride * N);
    double* const out_batch = out + (b * Stride * N);
    for (size_t i = 0; i < N; ++i) {
      std::array<double, Stride> accumulator{};
      for (size_t k = 0; k < N; ++k) {
        const double matrix_ik = matrix[i + (k * N)];
        const double* const u_slice = u_batch + (k * Stride);
        for (size_t x = 0; x < Stride; ++x) {
          accumulator[x] += matrix_ik * u_slice[x];
        }
      }
      double* const out_slice = out_batch + (i * Stride);
      for (size_t x = 0; x < Stride; ++x) {
        out_slice[x] = accumulator[x];
      }
    }
  }
}

// One register-resident strip of width StripWidth for the runtime-stride
// variant below: out_slice[s..s+W) = sum_k D(i,k) u(s.., k).
template <size_t N, size_t StripWidth>
void middle_dim_strip(double* const out_slice, const double* const u_batch,
                      const double* const matrix, const size_t i,
                      const size_t stride, const size_t s) {
  std::array<double, StripWidth> accumulator{};
  for (size_t k = 0; k < N; ++k) {
    const double matrix_ik = matrix[i + (k * N)];
    const double* const u_slice = u_batch + (k * stride) + s;
    for (size_t j = 0; j < StripWidth; ++j) {
      accumulator[j] += matrix_ik * u_slice[j];
    }
  }
  for (size_t j = 0; j < StripWidth; ++j) {
    out_slice[s + j] = accumulator[j];
  }
}

// Runtime-stride variant, used when the stride exceeds the dispatch range
// (i.e. for the slowest dimension of most meshes). Vectorized over the
// contiguous fastest index in register-resident strips; the remainder after
// the strips of 8 is handled by a 4/2/1 cascade of smaller strips so it stays
// vectorized.
template <size_t N>
void differentiate_middle_dim(double* const out, const double* const u,
                              const double* const matrix, const size_t stride,
                              const size_t number_of_batches) {
  for (size_t b = 0; b < number_of_batches; ++b) {
    const double* const u_batch = u + (b * stride * N);
    double* const out_batch = out + (b * stride * N);
    for (size_t i = 0; i < N; ++i) {
      double* const out_slice = out_batch + (i * stride);
      size_t s = 0;
      for (; s + 8 <= stride; s += 8) {
        middle_dim_strip<N, 8>(out_slice, u_batch, matrix, i, stride, s);
      }
      if (s + 4 <= stride) {
        middle_dim_strip<N, 4>(out_slice, u_batch, matrix, i, stride, s);
        s += 4;
      }
      if (s + 2 <= stride) {
        middle_dim_strip<N, 2>(out_slice, u_batch, matrix, i, stride, s);
        s += 2;
      }
      if (s < stride) {
        middle_dim_strip<N, 1>(out_slice, u_batch, matrix, i, stride, s);
      }
    }
  }
}

// Differentiate along a middle or the slowest logical dimension, using the
// compile-time-stride kernel whenever the stride is in dispatch range (always
// true for the middle dimension when the fast path is active).
template <size_t N>
void differentiate_middle_dim_dispatch(double* const out, const double* const u,
                                       const double* const matrix,
                                       const size_t stride,
                                       const size_t number_of_batches) {
  const bool used_small_stride =
      dispatch_on_extent(stride, [&](const auto stride_v) {
        differentiate_middle_dim_small_stride<N, decltype(stride_v)::value>(
            out, u, matrix, number_of_batches);
      });
  if (not used_small_stride) {
    differentiate_middle_dim<N>(out, u, matrix, stride, number_of_batches);
  }
}

template <size_t Dim>
bool fast_path_supports(const Mesh<Dim>& mesh) {
  for (size_t d = 0; d < Dim; ++d) {
    if (mesh.basis(d) != Spectral::Basis::Legendre and
        mesh.basis(d) != Spectral::Basis::Chebyshev) {
      return false;
    }
    if (mesh.extents(d) < 2 or mesh.extents(d) > fast_path_max_extent) {
      return false;
    }
  }
  return true;
}
}  // namespace

bool logical_derivatives_fast_path(
    const std::array<double*, 1>& logical_derivs, const double* const u,
    const size_t number_of_independent_components, const Mesh<1>& mesh) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  dispatch_on_extent(mesh.extents(0), [&](const auto extent) {
    differentiate_first_dim<decltype(extent)::value>(
        logical_derivs[0], u, matrix_xi.data(),
        number_of_independent_components);
  });
  return true;
}

bool logical_derivatives_fast_path(
    const std::array<double*, 2>& logical_derivs, const double* const u,
    const size_t number_of_independent_components, const Mesh<2>& mesh) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const size_t n0 = mesh.extents(0);
  const size_t n1 = mesh.extents(1);
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  const PackedMatrix matrix_eta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(1)));
  dispatch_on_extent(n0, [&](const auto extent) {
    differentiate_first_dim<decltype(extent)::value>(
        logical_derivs[0], u, matrix_xi.data(),
        number_of_independent_components * n1);
  });
  dispatch_on_extent(n1, [&](const auto extent) {
    differentiate_middle_dim_dispatch<decltype(extent)::value>(
        logical_derivs[1], u, matrix_eta.data(), n0,
        number_of_independent_components);
  });
  return true;
}

bool logical_derivatives_fast_path(
    const std::array<double*, 3>& logical_derivs, const double* const u,
    const size_t number_of_independent_components, const Mesh<3>& mesh) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const size_t n0 = mesh.extents(0);
  const size_t n1 = mesh.extents(1);
  const size_t n2 = mesh.extents(2);
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  const PackedMatrix matrix_eta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(1)));
  const PackedMatrix matrix_zeta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(2)));
  // All three kernels batch over every component at once; for eta the batch
  // also folds in the z-planes since the [comp][z] slabs are contiguous.
  dispatch_on_extent(n0, [&](const auto extent) {
    differentiate_first_dim<decltype(extent)::value>(
        logical_derivs[0], u, matrix_xi.data(),
        number_of_independent_components * n1 * n2);
  });
  dispatch_on_extent(n1, [&](const auto extent) {
    differentiate_middle_dim_dispatch<decltype(extent)::value>(
        logical_derivs[1], u, matrix_eta.data(), n0,
        number_of_independent_components * n2);
  });
  dispatch_on_extent(n2, [&](const auto extent) {
    differentiate_middle_dim_dispatch<decltype(extent)::value>(
        logical_derivs[2], u, matrix_zeta.data(), n0 * n1,
        number_of_independent_components);
  });
  return true;
}

bool fused_partial_derivatives_fast_path(
    double* const du, const double* const u,
    const size_t number_of_independent_components, const Mesh<1>& mesh,
    const std::array<const double*, 1>& inverse_jacobian) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const size_t number_of_points = mesh.extents(0);
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  DataVector buffer{number_of_points};
  double* const logical_xi = buffer.data();
  const double* const jacobian_xi = inverse_jacobian[0];
  for (size_t component = 0; component < number_of_independent_components;
       ++component) {
    const double* const u_component = u + (component * number_of_points);
    dispatch_on_extent(number_of_points, [&](const auto extent) {
      differentiate_first_dim<decltype(extent)::value>(logical_xi, u_component,
                                                       matrix_xi.data(), 1);
    });
    double* const du_component = du + (component * number_of_points);
    for (size_t p = 0; p < number_of_points; ++p) {
      du_component[p] = jacobian_xi[p] * logical_xi[p];
    }
  }
  return true;
}

bool fused_partial_derivatives_fast_path(
    double* const du, const double* const u,
    const size_t number_of_independent_components, const Mesh<2>& mesh,
    const std::array<const double*, 4>& inverse_jacobian) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const size_t n0 = mesh.extents(0);
  const size_t n1 = mesh.extents(1);
  const size_t number_of_points = n0 * n1;
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  const PackedMatrix matrix_eta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(1)));
  DataVector buffer{2 * number_of_points};
  double* const logical_xi = buffer.data();
  double* const logical_eta = buffer.data() + number_of_points;
  for (size_t component = 0; component < number_of_independent_components;
       ++component) {
    const double* const u_component = u + (component * number_of_points);
    dispatch_on_extent(n0, [&](const auto extent) {
      differentiate_first_dim<decltype(extent)::value>(logical_xi, u_component,
                                                       matrix_xi.data(), n1);
    });
    dispatch_on_extent(n1, [&](const auto extent) {
      differentiate_middle_dim_dispatch<decltype(extent)::value>(
          logical_eta, u_component, matrix_eta.data(), n0, 1);
    });
    for (size_t i = 0; i < 2; ++i) {
      double* const du_component =
          du + ((component * 2 + i) * number_of_points);
      const double* const jacobian_xi = inverse_jacobian[i];
      const double* const jacobian_eta = inverse_jacobian[2 + i];
      for (size_t p = 0; p < number_of_points; ++p) {
        du_component[p] = (jacobian_xi[p] * logical_xi[p]) +
                          (jacobian_eta[p] * logical_eta[p]);
      }
    }
  }
  return true;
}

bool fused_partial_derivatives_fast_path(
    double* const du, const double* const u,
    const size_t number_of_independent_components, const Mesh<3>& mesh,
    const std::array<const double*, 9>& inverse_jacobian) {
  if (not fast_path_supports(mesh)) {
    return false;
  }
  const size_t n0 = mesh.extents(0);
  const size_t n1 = mesh.extents(1);
  const size_t n2 = mesh.extents(2);
  const size_t number_of_points = n0 * n1 * n2;
  const PackedMatrix matrix_xi =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(0)));
  const PackedMatrix matrix_eta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(1)));
  const PackedMatrix matrix_zeta =
      pack_matrix(Spectral::differentiation_matrix(mesh.slice_through(2)));
  // Per-component logical-derivative buffers; small enough to stay resident
  // in the core-private cache while the Jacobian contraction consumes them.
  DataVector buffer{3 * number_of_points};
  double* const logical_xi = buffer.data();
  double* const logical_eta = buffer.data() + number_of_points;
  double* const logical_zeta = buffer.data() + (2 * number_of_points);
  for (size_t component = 0; component < number_of_independent_components;
       ++component) {
    const double* const u_component = u + (component * number_of_points);
    dispatch_on_extent(n0, [&](const auto extent) {
      differentiate_first_dim<decltype(extent)::value>(
          logical_xi, u_component, matrix_xi.data(), n1 * n2);
    });
    dispatch_on_extent(n1, [&](const auto extent) {
      differentiate_middle_dim_dispatch<decltype(extent)::value>(
          logical_eta, u_component, matrix_eta.data(), n0, n2);
    });
    dispatch_on_extent(n2, [&](const auto extent) {
      differentiate_middle_dim_dispatch<decltype(extent)::value>(
          logical_zeta, u_component, matrix_zeta.data(), n0 * n1, 1);
    });
    for (size_t i = 0; i < 3; ++i) {
      double* const du_component =
          du + ((component * 3 + i) * number_of_points);
      const double* const jacobian_xi = inverse_jacobian[i];
      const double* const jacobian_eta = inverse_jacobian[3 + i];
      const double* const jacobian_zeta = inverse_jacobian[6 + i];
      for (size_t p = 0; p < number_of_points; ++p) {
        du_component[p] = (jacobian_xi[p] * logical_xi[p]) +
                          (jacobian_eta[p] * logical_eta[p]) +
                          (jacobian_zeta[p] * logical_zeta[p]);
      }
    }
  }
  return true;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)
}  // namespace partial_derivatives_detail

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const gsl::not_null<gsl::span<typename DataType::value_type>*> buffer,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  static_assert(
      Dim > 0 and Dim < 4,
      "logical_partial_derivative is only implemented for 1, 2, and 3d");
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(buffer->size() >= num_grid_points,
         "The buffer in logical_partial_derivative must be at least of size "
             << num_grid_points << " but is of size " << buffer->size());

  set_number_of_grid_points(logical_derivative_of_u,
                            mesh.number_of_grid_points());
  if (Dim == 3 and mesh.basis(1) == Spectral::Basis::SphericalHarmonic) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      const Matrix& differentiation_matrix_xi =
          Spectral::differentiation_matrix(mesh.slice_through(0));
      const auto& ylm = ylm::get_spherepack_cache(mesh.extents(1) - 1);
      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 0_st)).data(),
            u[storage_index].data(), differentiation_matrix_xi,
            num_grid_points);
        const auto du = std::array{
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 1_st)).data(),
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 2_st)).data()};
        ylm.gradient_all_offsets(du, make_not_null(u[storage_index].data()),
                                 mesh.extents(0));
      }
    } else {
      ERROR(
          "Support for complex numbers with spherical harmonics is not yet "
          "implemented for logical_partial_derivative.");
    }
  } else if (Dim == 3 and mesh.basis(0) == Spectral::Basis::ZernikeB3) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      const size_t n_r = mesh.extents(0);
      const size_t l_max = mesh.extents(1) - 1;
      ASSERT(l_max < 2 * n_r - 1,
             "ZernikeB3 radial resolution is insufficient for the requested "
             "angular resolution. Need l_max < 2*n_r-1, but l_max="
                 << l_max << " and n_r=" << n_r);
      const auto& ylm = ylm::get_spherepack_cache(l_max);
      const size_t n_spectral = ylm.spectral_size();
      const size_t n_components = u.size();

      // Angular derivatives
      for (size_t storage_index = 0; storage_index < n_components;
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto du_ang = std::array{
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 1_st)).data(),
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(prepend(u_tensor_index, 2_st)).data()};
        ylm.gradient_all_offsets(du_ang, make_not_null(u[storage_index].data()),
                                 n_r);
      }

      // Radial derivatives, using parity-aware differentiation.
      // Functions on the ball with angular degree l behave as r^l near the
      // origin, giving even parity for even l and odd parity for odd l.
      const Matrix& D_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB3, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& D_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB3, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);

      // n_spectral is the SPHEREPACK buffer size = 2*(l_max+1)^2, including
      // padding. Valid modes total (l_max+1)^2; get_spherepack_cache always
      // sets m_max = l_max so each degree l contributes 2l+1 valid modes.
      const size_t n_valid_modes = (l_max + 1) * (l_max + 1);
      const size_t n_even_modes = (l_max % 2 == 0)
                                      ? (l_max + 1) * (l_max + 2) / 2
                                      : l_max * (l_max + 1) / 2;
      const size_t n_odd_modes = n_valid_modes - n_even_modes;

      const size_t spec_total = n_components * n_spectral * n_r;
      const size_t even_total = n_even_modes * n_components * n_r;
      const size_t odd_total = n_odd_modes * n_components * n_r;
      // NOLINTNEXTLINE(modernize-avoid-c-arrays)
      const auto all_buf = cpp20::make_unique_for_overwrite<double[]>(
          spec_total + 2 * (even_total + odd_total));
      double* const spec_buf = all_buf.get();
      double* const even_data = spec_buf + spec_total;      // NOLINT
      double* const odd_data = even_data + even_total;      // NOLINT
      double* const even_result = odd_data + odd_total;     // NOLINT
      double* const odd_result = even_result + even_total;  // NOLINT

      // SH physical -> spectral
      for (size_t n = 0; n < n_components; ++n) {
        ylm.phys_to_spec_all_offsets(
            make_not_null(spec_buf + n * n_spectral * n_r),  // NOLINT
            make_not_null(u[n].data()), n_r);
      }

      // Gather even-l and odd-l radial profiles into compact buffers.
      // Mode at SPHEREPACK offset s has its n_r values at
      // spec_buf[...+s*n_r].
      ylm::SpherepackIterator spec_iter{l_max, ylm.m_max()};
      for (size_t n = 0; n < n_components; ++n) {
        const double* spec = spec_buf + n * n_spectral * n_r;    // NOLINT
        double* even_dest = even_data + n * n_even_modes * n_r;  // NOLINT
        double* odd_dest = odd_data + n * n_odd_modes * n_r;     // NOLINT
        size_t even_col = 0;
        size_t odd_col = 0;
        spec_iter.reset();
        while (spec_iter) {
          const size_t s = spec_iter();
          if (spec_iter.l() % 2 == 0) {
            std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                      even_dest + even_col * n_r);           // NOLINT
            ++even_col;
          } else {
            std::copy(spec + s * n_r, spec + (s + 1) * n_r,  // NOLINT
                      odd_dest + odd_col * n_r);             // NOLINT
            ++odd_col;
          }
          ++spec_iter;
        }
      }

      // Apply radial differentiation matrices
      if (n_even_modes > 0) {
        partial_derivatives_detail::apply_matrix_in_first_dim(
            even_result, even_data, D_even, even_total);
      }
      if (n_odd_modes > 0) {
        partial_derivatives_detail::apply_matrix_in_first_dim(
            odd_result, odd_data, D_odd, odd_total);
      }

      // Copy differentiated radial profiles back to spectral buffer
      for (size_t n = 0; n < n_components; ++n) {
        double* spec = spec_buf + n * n_spectral * n_r;  // NOLINT
        const double* even_src =
            even_result + n * n_even_modes * n_r;                    // NOLINT
        const double* odd_src = odd_result + n * n_odd_modes * n_r;  // NOLINT
        size_t even_col = 0;
        size_t odd_col = 0;
        spec_iter.reset();
        while (spec_iter) {
          const size_t s = spec_iter();
          if (spec_iter.l() % 2 == 0) {
            std::copy(even_src + even_col * n_r,        // NOLINT
                      even_src + (even_col + 1) * n_r,  // NOLINT
                      spec + s * n_r);                  // NOLINT
            ++even_col;
          } else {
            std::copy(odd_src + odd_col * n_r,        // NOLINT
                      odd_src + (odd_col + 1) * n_r,  // NOLINT
                      spec + s * n_r);                // NOLINT
            ++odd_col;
          }
          ++spec_iter;
        }
      }

      // SH spectral -> physical
      for (size_t n = 0; n < n_components; ++n) {
        const auto u_tensor_index = u.get_tensor_index(n);
        ylm.spec_to_phys_all_offsets(
            make_not_null(
                // NOLINTNEXTLINE(readability-redundant-smartptr-get)
                logical_derivative_of_u->get(prepend(u_tensor_index, 0_st))
                    .data()),
            make_not_null(spec_buf + n * n_spectral * n_r), n_r);  // NOLINT
      }
    } else {
      ERROR(
          "Support for complex numbers with true filled sphere is not yet "
          "implemented for logical_partial_derivative.");
    }
  } else if (Dim == 2 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);
      const size_t M = n_phi / 2;

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);

        // Do phi derivative
        dgemm_<true>(
            'N',
            'T',  // transpose differentiation matrix to act on rows of data
            n_r, n_phi, n_phi, 1.0, u[storage_index].data(), n_r,
            diff_matrix_phi.data(), diff_matrix_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(phi_deriv_tensor_index).data(), n_r);

        // Get u in terms of r and m index
        dgemm_<true>('N', 'T', n_r, n_phi, n_phi, 1.0, u[storage_index].data(),
                     n_r, nodal_to_modal_phi.data(),
                     nodal_to_modal_phi.spacing(), 0.0,
                     // NOLINTNEXTLINE(readability-redundant-smartptr-get)
                     logical_derivative_of_u->get(r_deriv_tensor_index).data(),
                     n_r);  // C and ldc

        // Do radial derivative
        // m = 0
        dgemv_('N', n_r, n_r, 1.0, diff_matrix_r_even.data(),
               diff_matrix_r_even.spacing(),
               // NOLINTNEXTLINE(readability-redundant-smartptr-get)
               logical_derivative_of_u->get(r_deriv_tensor_index).data(), 1,
               0.0,  // beta = 0.0
               (*buffer).data(), 1);
        // j is index into Fourier modal vector
        // {u_0, u_1, u_{-1}, u_2, u_{-2}, ... , u_M, u_{-M}}
        size_t j = 1;
        for (size_t m = 1; m <= M; ++m) {
          const Matrix& diff_r =
              m % 2 == 0 ? diff_matrix_r_even : diff_matrix_r_odd;
          dgemm_<true>(
              'N', 'N', n_r, 2, n_r, 1.0, diff_r.data(), diff_r.spacing(),
              // NOLINTNEXTLINE(readability-redundant-smartptr-get)
              logical_derivative_of_u->get(r_deriv_tensor_index).data() +
                  j * n_r,
              n_r,
              0.0,  // beta = 0.0
              // NOLINTNEXTLINE
              (*buffer).data() + j * n_r, n_r);
          j += 2;
        }
        // Transform from m back to phi
        dgemm_<true>(
            'N', 'T', n_r, n_phi, n_phi, 1.0, (*buffer).data(), n_r,
            modal_to_nodal_phi.data(), modal_to_nodal_phi.spacing(), 0.0,
            // NOLINTNEXTLINE(readability-redundant-smartptr-get)
            logical_derivative_of_u->get(r_deriv_tensor_index).data(), n_r);
      }
    } else {
      ERROR(
          "Support for complex numbers with disk is not yet implemented for "
          "logical_partial_derivative.");
    }
  } else if (Dim == 3 and mesh.basis(0) == Spectral::Basis::ZernikeB2) {
    if constexpr (std::is_same_v<typename DataType::value_type, double>) {
      ASSERT(mesh.basis(0) == Spectral::Basis::ZernikeB2 and
                 mesh.basis(1) == Spectral::Basis::ZernikeB2,
             "Unexpected basis combination: " << mesh.basis());
      const size_t n_r = mesh.extents(0);
      const size_t n_phi = mesh.extents(1);
      const size_t n_z = mesh.extents(2);
      const size_t n_r_max = 2 * n_r - 2;
      ASSERT(
          n_phi % 2 == 1,
          "Fourier with an even number of grid points can be unstable due to "
          "the top derivative not being representable");
      ASSERT(n_phi / 2 <= n_r_max,
             "Zernike & Fourier on a disk have angular resolution limited by "
             "extents in both dimensions. We choose to enforce the restriction "
             "that the Fourier modal space is not larger than the Zernike "
             "angular capabilities (which would waste space and be physically "
             "ill-motivated).\nn_phi / 2 = "
                 << n_phi / 2 << ", Maximum from Zernike = " << n_r_max);

      const Matrix& diff_matrix_r_even = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Even);
      const Matrix& diff_matrix_r_odd = Spectral::differentiation_matrix<
          Spectral::Basis::ZernikeB2, Spectral::Quadrature::GaussRadauUpper>(
          n_r, Spectral::Parity::Odd);
      const Matrix& diff_matrix_phi = Spectral::differentiation_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& diff_matrix_z =
          Spectral::differentiation_matrix(mesh.slice_through(2));

      const Matrix& nodal_to_modal_phi = Spectral::nodal_to_modal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);
      const Matrix& modal_to_nodal_phi = Spectral::modal_to_nodal_matrix<
          Spectral::Basis::Fourier, Spectral::Quadrature::Equiangular>(n_phi);

      for (size_t storage_index = 0; storage_index < u.size();
           ++storage_index) {
        const DataType& u_component = u[storage_index];
        const auto u_tensor_index = u.get_tensor_index(storage_index);
        const auto r_deriv_tensor_index = prepend(u_tensor_index, 0_st);
        DataType& r_deriv_component =
            logical_derivative_of_u->get(r_deriv_tensor_index);
        const auto phi_deriv_tensor_index = prepend(u_tensor_index, 1_st);
        DataType& phi_deriv_component =
            logical_derivative_of_u->get(phi_deriv_tensor_index);
        const auto z_deriv_tensor_index = prepend(u_tensor_index, 2_st);
        DataType& z_deriv_component =
            logical_derivative_of_u->get(z_deriv_tensor_index);

        // phi derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r, n_phi * n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_phi,
            num_grid_points);
        raw_transpose(make_not_null(phi_deriv_component.data()), buffer->data(),
                      n_phi * n_z, n_r);

        // See comments in disk_apply() in the .tpp for considerations about
        // the following code. For this function, we don't have enough buffer
        // space to cleanly use the approach in the .tpp, so we use the
        // masking approach instead. Note that timing shows this to only be a
        // few percent slower

        // reuse transposed data to go to angular spectral space
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            nodal_to_modal_phi, num_grid_points);
        // r_deriv_component holds transposed angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_phi * n_z, n_r);
        std::copy(buffer->data(), buffer->data() + num_grid_points,
                  z_deriv_component.data());
        // buffer and z_deriv_component hold angular modal rep

        // buffer stores even components, z_deriv_component stores odd
        for (size_t k = 0; k < n_z; ++k) {
          size_t offset = k * n_phi * n_r;
          for (size_t i = 0; i < n_phi; ++i) {
            if (i == 0) {
              // This is an even column -> zero odd vals
              std::fill(z_deriv_component.data() + offset,
                        z_deriv_component.data() + offset + n_r, 0.0);
            } else {
              if ((i - 1) / 2 % 2 == 1) {
                // This is an even column with even next to it -> zero odd vals
                std::fill(z_deriv_component.data() + offset,
                          z_deriv_component.data() + offset + 2 * n_r, 0.0);
              } else {
                // This is an even column with odd next to it -> zero even vals
                std::fill(buffer->data() + offset,
                          buffer->data() + offset + 2 * n_r, 0.0);
              }
              ++i;
              offset += n_r;
            }
            offset += n_r;
          }
        }
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), buffer->data(), diff_matrix_r_even,
            num_grid_points, false);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            r_deriv_component.data(), z_deriv_component.data(),
            diff_matrix_r_odd, num_grid_points, true);
        // r_deriv_component holds the radial derivative in angular modal rep

        raw_transpose(make_not_null(buffer->data()), r_deriv_component.data(),
                      n_r, n_phi * n_z);
        // buffer holds transposed radial derivative in angular modal rep

        partial_derivatives_detail::apply_matrix_in_first_dim(
            z_deriv_component.data(), buffer->data(), modal_to_nodal_phi,
            num_grid_points);
        // z_deriv_component holds transposed radial derivative in angular nodal
        // rep

        raw_transpose(make_not_null(r_deriv_component.data()),
                      z_deriv_component.data(), n_phi * n_z, n_r);

        // z derivative
        raw_transpose(make_not_null(z_deriv_component.data()),
                      u_component.data(), n_r * n_phi, n_z);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), z_deriv_component.data(), diff_matrix_z,
            num_grid_points);
        raw_transpose(make_not_null(z_deriv_component.data()), buffer->data(),
                      n_z, n_r * n_phi);
      }
    } else {
      ERROR(
          "Support for complex numbers with cylinder is not yet implemented "
          "for logical_partial_derivative.");
    }
  } else {
    const Matrix empty_matrix{};
    std::array<std::reference_wrapper<const Matrix>, Dim> diff_matrices{
        make_array<Dim, std::reference_wrapper<const Matrix>>(empty_matrix)};
    for (size_t d = 0; d < Dim; ++d) {
      gsl::at(diff_matrices, d) =
          std::cref(Spectral::differentiation_matrix(mesh.slice_through(d)));
    }

    // It would be possible to check if the memory is contiguous and then
    // differentiate all components at once. Note that the buffer in that case
    // would also need to be the size of all components.
    for (size_t storage_index = 0; storage_index < u.size(); ++storage_index) {
      const auto u_tensor_index = u.get_tensor_index(storage_index);
      const auto xi_deriv_tensor_index = prepend(u_tensor_index, 0_st);
      partial_derivatives_detail::apply_matrix_in_first_dim(
          // NOLINTNEXTLINE(readability-redundant-smartptr-get)
          logical_derivative_of_u->get(xi_deriv_tensor_index).data(),
          u[storage_index].data(), diff_matrices[0].get(), num_grid_points);
      for (size_t i = 1; i < Dim; ++i) {
        const auto deriv_tensor_index = prepend(u_tensor_index, i);
        DataType& deriv_component =
            logical_derivative_of_u->get(deriv_tensor_index);
        size_t chunk_size =
            diff_matrices[0].get().rows() *
            (i == 1 ? 1 : gsl::at(diff_matrices, 1).get().rows());
        raw_transpose(make_not_null(deriv_component.data()),
                      u[storage_index].data(), chunk_size,
                      num_grid_points / chunk_size);
        partial_derivatives_detail::apply_matrix_in_first_dim(
            buffer->data(), deriv_component.data(),
            gsl::at(diff_matrices, i).get(), num_grid_points);
        chunk_size =
            i == 1 ? (Dim == 2 ? gsl::at(diff_matrices, 1).get().rows()
                               : gsl::at(diff_matrices, 1).get().rows() *
                                     gsl::at(diff_matrices, 2).get().rows())
                   : gsl::at(diff_matrices, 2).get().rows();
        raw_transpose(make_not_null(deriv_component.data()), buffer->data(),
                      chunk_size, num_grid_points / chunk_size);
      }
    }
  }
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
void logical_partial_derivative(
    gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>*>
        logical_derivative_of_u,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh) {
  using ValueType = typename DataType::value_type;  // double or complex<double>
  std::vector<ValueType> buffer(mesh.number_of_grid_points());
  gsl::span<ValueType> buffer_view{buffer.data(), buffer.size()};
  logical_partial_derivative(logical_derivative_of_u,
                             make_not_null(&buffer_view), u, mesh);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim>
auto logical_partial_derivative(const Tensor<DataType, SymmList, IndexList>& u,
                                const Mesh<Dim>& mesh)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      result{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&result), u, mesh);
  return result;
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
        Frame::ElementLogical>& logical_partial_derivative_of_u,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  for (size_t storage_index = 0;
       storage_index < Tensor<DataType, SymmList, IndexList>::size();
       ++storage_index) {
    const auto u_multi_index =
        Tensor<DataType, SymmList,
               IndexList>::structure::get_canonical_tensor_index(storage_index);
    for (size_t i = 0; i < Dim; i++) {
      const auto du_multi_index = prepend(u_multi_index, i);
      du->get(du_multi_index) =
          inverse_jacobian.get(0, i) *
          logical_partial_derivative_of_u.get(prepend(u_multi_index, 0_st));
      for (size_t j = 1; j < Dim; j++) {
        du->get(du_multi_index) +=
            inverse_jacobian.get(j, i) *
            logical_partial_derivative_of_u.get(prepend(u_multi_index, j));
      }
    }
  }
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
void partial_derivative(
    const gsl::not_null<TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>*>
        du,
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian) {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo,
      Frame::ElementLogical>
      logical_partial_derivative_of_u{mesh.number_of_grid_points()};
  logical_partial_derivative(make_not_null(&logical_partial_derivative_of_u), u,
                             mesh);
  partial_derivative<DataType, SymmList, IndexList>(
      du, logical_partial_derivative_of_u, inverse_jacobian);
}

template <typename DataType, typename SymmList, typename IndexList, size_t Dim,
          typename DerivativeFrame>
auto partial_derivative(
    const Tensor<DataType, SymmList, IndexList>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          DerivativeFrame>& inverse_jacobian)
    -> TensorMetafunctions::prepend_spatial_index<
        Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame> {
  TensorMetafunctions::prepend_spatial_index<
      Tensor<DataType, SymmList, IndexList>, Dim, UpLo::Lo, DerivativeFrame>
      result{mesh.number_of_grid_points()};
  partial_derivative(make_not_null(&result), u, mesh, inverse_jacobian);
  return result;
}

#define GET_DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)
#define GET_TENSOR(data) BOOST_PP_TUPLE_ELEM(3, data)

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
                        GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),     \
                        GET_FRAME(data)>,                                      \
                    GET_DIM(data), UpLo::Lo, Frame::ElementLogical>* >         \
          logical_derivative_of_u,                                             \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh);                  \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      Frame::ElementLogical >                                                  \
          logical_partial_derivative(const GET_TENSOR(data) < GET_DTYPE(data), \
                                     GET_DIM(data), GET_FRAME(data) > &u,      \
                                     const Mesh<GET_DIM(data)>& mesh);         \
  template void                                                                \
      partial_derivative<GET_DTYPE(data), GET_TENSOR(data) < GET_DTYPE(data),  \
                         GET_DIM(data), GET_FRAME(data)>::symmetry,            \
      GET_TENSOR(                                                              \
          data)<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>::index_list > \
          (const gsl::not_null<TensorMetafunctions::prepend_spatial_index<     \
                                   GET_TENSOR(data) < GET_DTYPE(data),         \
                                   GET_DIM(data), GET_FRAME(data)>,            \
                               GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* >    \
               du,                                                             \
           const TensorMetafunctions::prepend_spatial_index<                   \
               GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),              \
               GET_FRAME(data)>,                                               \
           GET_DIM(data), UpLo::Lo,                                            \
           Frame::ElementLogical > &logical_partial_derivative_of_u,           \
           const InverseJacobian<DataVector, GET_DIM(data),                    \
                                 Frame::ElementLogical, GET_FRAME(data)>       \
               & inverse_jacobian);                                            \
  template void partial_derivative(                                            \
      const gsl::not_null<TensorMetafunctions::prepend_spatial_index<          \
                              GET_TENSOR(data) < GET_DTYPE(data),              \
                              GET_DIM(data), GET_FRAME(data)>,                 \
                          GET_DIM(data), UpLo::Lo, GET_FRAME(data)>* > du,     \
      const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                 \
      GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,                   \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template TensorMetafunctions::prepend_spatial_index<                         \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>,     \
      GET_DIM(data), UpLo::Lo,                                                 \
      GET_FRAME(data) >                                                        \
          partial_derivative(                                                  \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),         \
              GET_FRAME(data) > &u, const Mesh<GET_DIM(data)>& mesh,           \
              const InverseJacobian<DataVector, GET_DIM(data),                 \
                                    Frame::ElementLogical, GET_FRAME(data)>&   \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (tnsr::a, tnsr::A, tnsr::i, tnsr::I, tnsr::ab, tnsr::Ab,
                         tnsr::aB, tnsr::AB, tnsr::ij, tnsr::iJ, tnsr::Ij,
                         tnsr::IJ, tnsr::iA, tnsr::ia, tnsr::aa, tnsr::AA,
                         tnsr::ii, tnsr::II, tnsr::ijj, tnsr::Ijj, tnsr::iaa))

#undef INSTANTIATION

// Some additional mixed-dimension instantiations
template TensorMetafunctions::prepend_spatial_index<
    tnsr::aa<ComplexDataVector, 3, Frame::Inertial>, 2, UpLo::Lo,
    Frame::Inertial>
partial_derivative(const tnsr::aa<ComplexDataVector, 3, Frame::Inertial>& u,
                   const Mesh<2>& mesh,
                   const InverseJacobian<DataVector, 2, Frame::ElementLogical,
                                         Frame::Inertial>& inverse_jacobian);

#define INSTANTIATION(r, data)                                                 \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      gsl::not_null<gsl::span<typename GET_DTYPE(data)::value_type>*> buffer,  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template void logical_partial_derivative(                                    \
      gsl::not_null<TensorMetafunctions::prepend_spatial_index<                \
          Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo,                    \
          Frame::ElementLogical>*>                                             \
          logical_derivative_of_u,                                             \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh);      \
  template TensorMetafunctions::prepend_spatial_index<                         \
      Scalar<GET_DTYPE(data)>, GET_DIM(data), UpLo::Lo, Frame::ElementLogical> \
  logical_partial_derivative(const Scalar<GET_DTYPE(data)>& u,                 \
                             const Mesh<GET_DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (DataVector, ComplexDataVector),
                        (1, 2, 3))

#undef INSTANTIATION

#define INSTANTIATE_JACOBIANS(r, data)                                       \
  template TensorMetafunctions::prepend_spatial_index<                       \
      GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),                     \
      Frame::ElementLogical, GET_FRAME(data)>,                               \
      GET_DIM(data), UpLo::Lo,                                               \
      GET_FRAME(data) >                                                      \
          partial_derivative(                                                \
              const GET_TENSOR(data) < GET_DTYPE(data), GET_DIM(data),       \
              Frame::ElementLogical, GET_FRAME(data) > &u,                   \
              const Mesh<GET_DIM(data)>& mesh,                               \
              const InverseJacobian<DataVector, GET_DIM(data),               \
                                    Frame::ElementLogical, GET_FRAME(data)>& \
                  inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_JACOBIANS, (DataVector), (1, 2, 3),
                        (Frame::Inertial), (InverseJacobian))

#undef INSTANTIATE_JACOBIANS

#define INSTANTIATE_SCALAR(r, data)                                            \
  template void partial_derivative<GET_DTYPE(data), Symmetry<>, index_list<>>( \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const tnsr::i<GET_DTYPE(data), GET_DIM(data), Frame::ElementLogical>&    \
          logical_partial_derivative_of_u,                                     \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template void partial_derivative(                                            \
      gsl::not_null<tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>*> \
          du,                                                                  \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);               \
  template tnsr::i<GET_DTYPE(data), GET_DIM(data), GET_FRAME(data)>            \
  partial_derivative(                                                          \
      const Scalar<GET_DTYPE(data)>& u, const Mesh<GET_DIM(data)>& mesh,       \
      const InverseJacobian<DataVector, GET_DIM(data), Frame::ElementLogical,  \
                            GET_FRAME(data)>& inverse_jacobian);

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR, (DataVector, ComplexDataVector),
                        (1, 2, 3),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef INSTANTIATE_SCALAR
#undef GET_FRAME
#undef GET_DIM
#undef GET_TENSOR
