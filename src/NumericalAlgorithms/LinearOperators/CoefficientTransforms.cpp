// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/CoefficientTransforms.hpp"

#include <array>
#include <utility>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"  // IWYU pragma: keep
#include "DataStructures/ModalVector.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_forward_declare Matrix

namespace {
template <size_t Dim, size_t... Is>
std::array<std::reference_wrapper<const Matrix>, Dim> make_transform_matrices(
    const Mesh<Dim>& mesh, const bool nodal_to_modal,
    std::index_sequence<Is...> /*meta*/) {
  return nodal_to_modal ? std::array<std::reference_wrapper<const Matrix>,
                                     Dim>{{Spectral::nodal_to_modal_matrix(
                              mesh.slice_through(Is))...}}
                        : std::array<std::reference_wrapper<const Matrix>, Dim>{
                              {Spectral::modal_to_nodal_matrix(
                                  mesh.slice_through(Is))...}};
}
}  // namespace

template <size_t Dim>
void to_modal_coefficients(
    const gsl::not_null<ComplexModalVector*> modal_coefficients,
    const ComplexDataVector& nodal_coefficients, const Mesh<Dim>& mesh) {
  modal_coefficients->destructive_resize(nodal_coefficients.size());
  apply_matrices<ComplexModalVector>(
      modal_coefficients,
      make_transform_matrices<Dim>(mesh, true, std::make_index_sequence<Dim>{}),
      nodal_coefficients, mesh.extents());
}

// overload provided so that the most common case of transforming from
// `DataVector` to `ModalVector` does not require additional `make_not_null`s
template <size_t Dim>
void to_modal_coefficients(gsl::not_null<ModalVector*> modal_coefficients,
                           const DataVector& nodal_coefficients,
                           const Mesh<Dim>& mesh) {
  modal_coefficients->destructive_resize(nodal_coefficients.size());
  apply_matrices<ModalVector>(
      modal_coefficients,
      make_transform_matrices<Dim>(mesh, true, std::make_index_sequence<Dim>{}),
      nodal_coefficients, mesh.extents());
}

template <size_t Dim>
ModalVector to_modal_coefficients(const DataVector& nodal_coefficients,
                                  const Mesh<Dim>& mesh) {
  ModalVector modal_coefficients(nodal_coefficients.size());
  to_modal_coefficients(make_not_null(&modal_coefficients), nodal_coefficients,
                        mesh);
  return modal_coefficients;
}

template <size_t Dim>
ComplexModalVector to_modal_coefficients(
    const ComplexDataVector& nodal_coefficients, const Mesh<Dim>& mesh) {
  ComplexModalVector modal_coefficients(nodal_coefficients.size());
  to_modal_coefficients(make_not_null(&modal_coefficients), nodal_coefficients,
                        mesh);
  return modal_coefficients;
}

template <size_t Dim>
void to_nodal_coefficients(
    const gsl::not_null<ComplexDataVector*> nodal_coefficients,
    const ComplexModalVector& modal_coefficients, const Mesh<Dim>& mesh) {
  nodal_coefficients->destructive_resize(modal_coefficients.size());
  apply_matrices<ComplexDataVector>(
      nodal_coefficients,
      make_transform_matrices<Dim>(mesh, false,
                                   std::make_index_sequence<Dim>{}),
      modal_coefficients, mesh.extents());
}

template <size_t Dim>
void to_nodal_coefficients(const gsl::not_null<DataVector*> nodal_coefficients,
                           const ModalVector& modal_coefficients,
                           const Mesh<Dim>& mesh) {
  nodal_coefficients->destructive_resize(modal_coefficients.size());
  apply_matrices<DataVector>(nodal_coefficients,
                             make_transform_matrices<Dim>(
                                 mesh, false, std::make_index_sequence<Dim>{}),
                             modal_coefficients, mesh.extents());
}

template <size_t Dim>
DataVector to_nodal_coefficients(const ModalVector& modal_coefficients,
                                 const Mesh<Dim>& mesh) {
  DataVector nodal_coefficients(modal_coefficients.size());
  to_nodal_coefficients(make_not_null(&nodal_coefficients), modal_coefficients,
                        mesh);
  return nodal_coefficients;
}

template <size_t Dim>
ComplexDataVector to_nodal_coefficients(
    const ComplexModalVector& modal_coefficients, const Mesh<Dim>& mesh) {
  ComplexDataVector nodal_coefficients(modal_coefficients.size());
  to_nodal_coefficients(make_not_null(&nodal_coefficients), modal_coefficients,
                        mesh);
  return nodal_coefficients;
}

namespace {
template <typename Type>
struct modal_type_to_nodal_type;

template <>
struct modal_type_to_nodal_type<ModalVector> {
  using type = DataVector;
};

template <>
struct modal_type_to_nodal_type<ComplexModalVector> {
  using type = ComplexDataVector;
};
}  // namespace

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_TYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_TO_MODAL_COEFFICIENTS(r, data)                   \
  template void to_modal_coefficients(                               \
      const gsl::not_null<GET_TYPE(data)*> modal_coefficients,       \
      const typename modal_type_to_nodal_type<GET_TYPE(data)>::type& \
          nodal_coefficients,                                        \
      const Mesh<GET_DIM(data)>& mesh);                              \
  template GET_TYPE(data) to_modal_coefficients(                     \
      const typename modal_type_to_nodal_type<GET_TYPE(data)>::type& \
          nodal_coefficients,                                        \
      const Mesh<GET_DIM(data)>& mesh);

#define INSTANTIATE_TO_NODAL_COEFFICIENTS(r, data)                  \
  template void to_nodal_coefficients(                              \
      const gsl::not_null<                                          \
          typename modal_type_to_nodal_type<GET_TYPE(data)>::type*> \
          nodal_coefficients,                                       \
      const GET_TYPE(data) & modal_coefficients,                    \
      const Mesh<GET_DIM(data)>& mesh);                             \
  template typename modal_type_to_nodal_type<GET_TYPE(data)>::type  \
  to_nodal_coefficients(const GET_TYPE(data) & modal_coefficients,  \
                        const Mesh<GET_DIM(data)>& mesh);

GENERATE_INSTANTIATIONS(INSTANTIATE_TO_MODAL_COEFFICIENTS, (1, 2, 3),
                        (ModalVector, ComplexModalVector))

GENERATE_INSTANTIATIONS(INSTANTIATE_TO_NODAL_COEFFICIENTS, (1, 2, 3),
                        (ModalVector, ComplexModalVector))

#undef GET_DIM
#undef GET_TYPE
#undef INSTANTIATE_TO_NODAL_COEFFICIENTS
#undef INSTANTIATE_TO_MODAL_COEFFICIENTS
