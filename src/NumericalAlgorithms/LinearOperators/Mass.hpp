// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions and tags for taking a divergence.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
template <size_t Dim>
class Mesh;
/// \endcond

namespace Tags {

template <typename Tag>
struct Mass : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = tmpl::type_from<Tag>;
};

}  // namespace Tags

// TODO: resolve jacobians better with higher-order quadrature

template <typename VariablesTags, size_t Dim>
Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> mass(
    const Variables<VariablesTags>& variables, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& det_jacobian) noexcept {
  std::array<Matrix, Dim> mass_matrices{};
  for (size_t d = 0; d < Dim; d++) {
    gsl::at(mass_matrices, d) = Spectral::mass_matrix(mesh.slice_through(d));
  }
  return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(apply_matrices(
      mass_matrices, variables * get(det_jacobian), mesh.extents()));
}

template <typename VariablesTags, size_t Dim, typename IntegrationFrame>
Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> inv_mass(
    const Variables<VariablesTags>& variables, const Mesh<Dim>& mesh,
    const Scalar<DataVector>& det_inv_jacobian) noexcept {
  std::array<Matrix, Dim> inv_mass_matrices{};
  for (size_t d = 0; d < Dim; d++) {
    // TODO: Cache inverse
    gsl::at(inv_mass_matrices, d) =
        blaze::inv(Spectral::mass_matrix(mesh.slice_through(d)));
  }
  return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(apply_matrices(
      inv_mass_matrices, variables * get(det_inv_jacobian), mesh.extents()));
}

template <size_t FaceDim, typename VariablesTags>
Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>> mass_on_face(
    const Variables<VariablesTags>& vars_on_face,
    const Mesh<FaceDim>& face_mesh,
    const Scalar<DataVector>& surface_jacobian) noexcept {
  if constexpr (FaceDim == 0) {
    return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(vars_on_face);
  } else {
    std::array<Matrix, FaceDim> mass_matrices{};
    for (size_t d = 0; d < FaceDim; d++) {
      gsl::at(mass_matrices, d) =
          Spectral::mass_matrix(face_mesh.slice_through(d));
    }
    return Variables<db::wrap_tags_in<Tags::Mass, VariablesTags>>(
        apply_matrices(mass_matrices, vars_on_face * get(surface_jacobian),
                       face_mesh.extents()));
  }
}

/*
template <typename VariablesTags, size_t VolumeDim, typename DiffFrame>
Variables<db::wrap_tags_in<::Tags::div, VariablesTags>> stiffness(
    const Variables<VariablesTags>& variables, const Mesh<VolumeDim>& mesh,
    const InverseJacobian<DataVector, VolumeDim, Frame::Logical, DiffFrame>&
        inv_jacobian) noexcept {
  Variables<db::wrap_tags_in<::Tags::div, VariablesTags>> result{
      variables.number_of_grid_points(), 0.};
  for (size_t d = 0; d < VolumeDim; d++) {
    std::array<Matrix, VolumeDim> diff_matrices_transpose{};
    Matrix& diff_matrix = gsl::at(diff_matrices_transpose, d);
    diff_matrix =
        trans(Spectral::differentiation_matrix(mesh.slice_through(d)));
    // // Jacobian is diagonal
    // const DataVector& jacobian_factor = inv_jacobian.get(d, d);
    // for (size_t i = 0; i < jacobian_factor.size(); i++) {
    //   for (size_t j = 0; j < jacobian_factor.size(); j++) {
    //     diff_matrix(i, j) *= jacobian_factor[j];
    //   }
    // }
    // We only need the d-th first-index component of the tensors in
    // `variables`, so instead of taking that in the loop below we could take it
    // here already, could speed this up
    auto derivs_this_dim =
        apply_matrices(diff_matrices_transpose, variables, mesh.extents());

    tmpl::for_each<VariablesTags>([&result, &derivs_this_dim, &d,
                                   &inv_jacobian](auto deriv_tag_v) noexcept {
      using deriv_tag = tmpl::type_from<decltype(deriv_tag_v)>;
      using div_tag = ::Tags::div<deriv_tag>;

      using first_index =
          tmpl::front<typename db::item_type<deriv_tag>::index_list>;
      static_assert(
          std::is_same_v<typename first_index::Frame, DiffFrame> and
              first_index::ul == UpLo::Up,
          "First index of tensor cannot be contracted with derivative "
          "because either it is in the wrong frame or it has the wrong "
          "valence");

      auto& div = get<div_tag>(result);
      for (auto it = div.begin(); it != div.end(); ++it) {
        const auto div_indices = div.get_tensor_index(it);
        const auto flux_indices = prepend(div_indices, d);
        // Jacobian is diagonal
        *it += inv_jacobian.get(d, d) *
               get<deriv_tag>(derivs_this_dim).get(flux_indices);
      }
    });
  }
  return result;
}
*/
