// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"

#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

template <typename ResultTensor, typename FluxTensor, size_t Dim>
void logical_weak_divergence(const gsl::not_null<ResultTensor*> div_flux,
                             const FluxTensor& flux, const Mesh<Dim>& mesh) {
  // Note: This function hasn't been optimized much at all. Feel free to
  // optimize if needed!
  static const Matrix identity_matrix{};
  for (size_t d = 0; d < Dim; ++d) {
    auto matrices = make_array<Dim>(std::cref(identity_matrix));
    gsl::at(matrices, d) =
        Spectral::differentiation_matrix_transpose(mesh.slice_through(d));
    for (size_t storage_index = 0; storage_index < div_flux->size();
         ++storage_index) {
      const auto div_flux_index = div_flux->get_tensor_index(storage_index);
      const auto flux_index = prepend(div_flux_index, d);
      div_flux->get(div_flux_index) +=
          apply_matrices(matrices, flux.get(flux_index), mesh.extents());
    }
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TENSOR(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION_SCALAR(r, data)                                    \
  template void logical_weak_divergence(                                 \
      const gsl::not_null<Scalar<DataVector>*> div_flux,                 \
      const tnsr::I<DataVector, DIM(data), Frame::ElementLogical>& flux, \
      const Mesh<DIM(data)>& mesh);
#define INSTANTIATION_TENSOR(r, data)                                   \
  template void logical_weak_divergence(                                \
      const gsl::not_null<tnsr::TENSOR(data) < DataVector, DIM(data),   \
                          Frame::Inertial>* > div_flux,                 \
      const TensorMetafunctions::prepend_spatial_index<                 \
          tnsr::TENSOR(data) < DataVector, DIM(data), Frame::Inertial>, \
      DIM(data), UpLo::Up, Frame::ElementLogical > &flux,               \
      const Mesh<DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION_SCALAR, (1, 2, 3))
GENERATE_INSTANTIATIONS(INSTANTIATION_TENSOR, (1, 2, 3), (i, I))

#undef INSTANTIATION
