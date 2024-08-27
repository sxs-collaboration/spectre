// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"

#include <complex>
#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace dg::detail {

template <typename ValueType, size_t Dim>
void apply_mass_matrix_impl(const gsl::not_null<ValueType*> data,
                            const Mesh<Dim>& mesh) {
  if constexpr (Dim == 0) {
    // Nothing to do
    (void)data;
    (void)mesh;
  } else if constexpr (Dim == 1) {
    const size_t x_size = mesh.extents(0);
    const auto& w_x = Spectral::quadrature_weights(mesh);
    for (size_t i = 0; i < x_size; ++i) {
      data.get()[i] *= w_x[i];
    }
  } else if constexpr (Dim == 2) {
    const size_t x_size = mesh.extents(0);
    const size_t y_size = mesh.extents(1);
    const auto& w_x = Spectral::quadrature_weights(mesh.slice_through(0));
    const auto& w_y = Spectral::quadrature_weights(mesh.slice_through(1));
    for (size_t j = 0; j < y_size; ++j) {
      const size_t offset = j * x_size;
      for (size_t i = 0; i < x_size; ++i) {
        data.get()[offset + i] *= w_x[i] * w_y[j];
      }
    }
  } else if constexpr (Dim == 3) {
    const size_t x_size = mesh.extents(0);
    const size_t y_size = mesh.extents(1);
    const size_t z_size = mesh.extents(2);
    const auto& w_x = Spectral::quadrature_weights(mesh.slice_through(0));
    const auto& w_y = Spectral::quadrature_weights(mesh.slice_through(1));
    const auto& w_z = Spectral::quadrature_weights(mesh.slice_through(2));
    for (size_t k = 0; k < z_size; ++k) {
      const size_t offset_z = k * y_size * x_size;
      for (size_t j = 0; j < y_size; ++j) {
        const double w_yz = w_y[j] * w_z[k];
        const size_t offset = x_size * j + offset_z;
        for (size_t i = 0; i < x_size; ++i) {
          data.get()[offset + i] *= w_x[i] * w_yz;
        }
      }
    }
  }
}

template <typename ValueType, size_t Dim>
void apply_inverse_mass_matrix_impl(const gsl::not_null<ValueType*> data,
                                    const Mesh<Dim>& mesh) {
  if constexpr (Dim == 0) {
    // Nothing to do
    (void)data;
    (void)mesh;
  } else if constexpr (Dim == 1) {
    const size_t x_size = mesh.extents(0);
    const auto& w_x = Spectral::quadrature_weights(mesh);
    for (size_t i = 0; i < x_size; ++i) {
      data.get()[i] /= w_x[i];
    }
  } else if constexpr (Dim == 2) {
    const size_t x_size = mesh.extents(0);
    const size_t y_size = mesh.extents(1);
    const auto& w_x = Spectral::quadrature_weights(mesh.slice_through(0));
    const auto& w_y = Spectral::quadrature_weights(mesh.slice_through(1));
    for (size_t j = 0; j < y_size; ++j) {
      const size_t offset = j * x_size;
      for (size_t i = 0; i < x_size; ++i) {
        data.get()[offset + i] /= w_x[i] * w_y[j];
      }
    }
  } else if constexpr (Dim == 3) {
    const size_t x_size = mesh.extents(0);
    const size_t y_size = mesh.extents(1);
    const size_t z_size = mesh.extents(2);
    const auto& w_x = Spectral::quadrature_weights(mesh.slice_through(0));
    const auto& w_y = Spectral::quadrature_weights(mesh.slice_through(1));
    const auto& w_z = Spectral::quadrature_weights(mesh.slice_through(2));
    for (size_t k = 0; k < z_size; ++k) {
      const size_t offset_z = k * y_size * x_size;
      for (size_t j = 0; j < y_size; ++j) {
        const double w_yz = w_y[j] * w_z[k];
        const size_t offset = x_size * j + offset_z;
        for (size_t i = 0; i < x_size; ++i) {
          data.get()[offset + i] /= w_x[i] * w_yz;
        }
      }
    }
  }
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void apply_mass_matrix_impl(gsl::not_null<DTYPE(data)*>,         \
                                       const Mesh<DIM(data)>&);             \
  template void apply_inverse_mass_matrix_impl(gsl::not_null<DTYPE(data)*>, \
                                               const Mesh<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, std::complex<double>),
                        (0, 1, 2, 3))

#undef DTYPE
#undef DIM
#undef INSTANTIATE

}  // namespace dg::detail
