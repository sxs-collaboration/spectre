// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"

#include <cstddef>

#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace dg::detail {

template <>
void apply_mass_matrix_impl<1>(const gsl::not_null<double*> data,
                               const Mesh<1>& mesh) {
  const size_t x_size = mesh.extents(0);
  const auto& w_x = Spectral::quadrature_weights(mesh);
  for (size_t i = 0; i < x_size; ++i) {
    data.get()[i] *= w_x[i];
  }
}

template <>
void apply_mass_matrix_impl<2>(const gsl::not_null<double*> data,
                               const Mesh<2>& mesh) {
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
}

template <>
void apply_mass_matrix_impl<3>(const gsl::not_null<double*> data,
                               const Mesh<3>& mesh) {
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

}  // namespace dg::detail
