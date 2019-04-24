// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/StripeIterator.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void transform_impl_helper(double* const output_coeffs,
                           const double* const input_coeffs,
                           const Mesh<Dim>& mesh, const bool nodal_to_modal,
                           const size_t dim) noexcept {
  for (StripeIterator stripe_it(mesh.extents(), dim); stripe_it; ++stripe_it) {
    const Matrix& transformation_matrix =
        nodal_to_modal
            ? Spectral::nodal_to_modal_matrix(mesh.slice_through(dim))
            : Spectral::modal_to_nodal_matrix(mesh.slice_through(dim));
    dgemv_('N', mesh.extents()[dim], mesh.extents()[dim], 1.0,
           transformation_matrix.data(), mesh.extents()[dim],
           input_coeffs + stripe_it.offset(), stripe_it.stride(),  // NOLINT
           0.0, output_coeffs + stripe_it.offset(),                // NOLINT
           stripe_it.stride());
  }
}

void transform_impl(double* const modal_coefficients,
                    const double* const nodal_coefficients, const Mesh<1>& mesh,
                    const bool nodal_to_modal) noexcept {
  transform_impl_helper(modal_coefficients, nodal_coefficients, mesh,
                        nodal_to_modal, 0);
}

void transform_impl(double* const modal_coefficients,
                    const double* const nodal_coefficients, const Mesh<2>& mesh,
                    const bool nodal_to_modal) noexcept {
  DataVector temp(mesh.number_of_grid_points());
  transform_impl_helper(temp.data(), nodal_coefficients, mesh, nodal_to_modal,
                        0);
  transform_impl_helper(modal_coefficients, temp.data(), mesh, nodal_to_modal,
                        1);
}

void transform_impl(double* const modal_coefficients,
                    const double* const nodal_coefficients, const Mesh<3>& mesh,
                    const bool nodal_to_modal) noexcept {
  DataVector temp(mesh.number_of_grid_points());
  transform_impl_helper(modal_coefficients, nodal_coefficients, mesh,
                        nodal_to_modal, 0);
  transform_impl_helper(temp.data(), modal_coefficients, mesh, nodal_to_modal,
                        1);
  transform_impl_helper(modal_coefficients, temp.data(), mesh, nodal_to_modal,
                        2);
}
}  // namespace

template <size_t Dim>
void to_modal_coefficients(const gsl::not_null<ModalVector*> modal_coefficients,
                           const DataVector& nodal_coefficients,
                           const Mesh<Dim>& mesh) noexcept {
  if (modal_coefficients->size() != nodal_coefficients.size()) {
    ASSERT(modal_coefficients->is_owning(),
           "Cannot resize a non-owning ModalVector");
    *modal_coefficients = ModalVector(nodal_coefficients.size());
  }
  transform_impl(modal_coefficients->data(), nodal_coefficients.data(), mesh,
                 true);
}

template <size_t Dim>
ModalVector to_modal_coefficients(const DataVector& nodal_coefficients,
                                  const Mesh<Dim>& mesh) noexcept {
  ModalVector modal_coefficients(nodal_coefficients.size());
  transform_impl(modal_coefficients.data(), nodal_coefficients.data(), mesh,
                 true);
  return modal_coefficients;
}

template <size_t Dim>
void to_nodal_coefficients(const gsl::not_null<DataVector*> nodal_coefficients,
                           const ModalVector& modal_coefficients,
                           const Mesh<Dim>& mesh) noexcept {
  if (nodal_coefficients->size() != modal_coefficients.size()) {
    ASSERT(nodal_coefficients->is_owning(),
           "Cannot resize a non-owning DataVector");
    *nodal_coefficients = DataVector(modal_coefficients.size());
  }
  transform_impl(nodal_coefficients->data(), modal_coefficients.data(), mesh,
                 false);
}

template <size_t Dim>
DataVector to_nodal_coefficients(const ModalVector& modal_coefficients,
                                 const Mesh<Dim>& mesh) noexcept {
  DataVector nodal_coefficients(modal_coefficients.size());
  transform_impl(nodal_coefficients.data(), modal_coefficients.data(), mesh,
                 false);
  return nodal_coefficients;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(r, data)                                 \
  template void to_modal_coefficients<GET_DIM(data)>(        \
      const gsl::not_null<ModalVector*> modal_coefficients,  \
      const DataVector& nodal_coefficients,                  \
      const Mesh<GET_DIM(data)>& mesh) noexcept;             \
  template ModalVector to_modal_coefficients<GET_DIM(data)>( \
      const DataVector& nodal_coefficients,                  \
      const Mesh<GET_DIM(data)>& mesh) noexcept;             \
  template void to_nodal_coefficients<GET_DIM(data)>(        \
      const gsl::not_null<DataVector*> nodal_coefficients,   \
      const ModalVector& modal_coefficients,                 \
      const Mesh<GET_DIM(data)>& mesh) noexcept;             \
  template DataVector to_nodal_coefficients<GET_DIM(data)>(  \
      const ModalVector& modal_coefficients,                 \
      const Mesh<GET_DIM(data)>& mesh) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))
#undef GET_DIM
#undef INSTANTIATE
