// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdArrayHelpers.hpp"

namespace partial_derivatives_detail {
template <size_t Dim, typename VariableTags, typename DerivativeTags>
struct LogicalImpl;
}  // namespace partial_derivatives_detail

template <typename DerivativeTags, typename VariableTags, size_t Dim>
std::array<Variables<DerivativeTags>, Dim> logical_partial_derivatives(
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh) noexcept {
  return partial_derivatives_detail::LogicalImpl<
      Dim, VariableTags, DerivativeTags>::apply(u, mesh);
}

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                           DerivativeFrame>>
partial_derivatives(
    const Variables<VariableTags>& u, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept {
  const auto logical_partial_derivatives_of_u =
      logical_partial_derivatives<DerivativeTags>(u, mesh);

  Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                             DerivativeFrame>>
      partial_derivatives_of_u(u.number_of_grid_points(), 0.0);

  tmpl::for_each<DerivativeTags>([
    &partial_derivatives_of_u, &inverse_jacobian,
    &logical_partial_derivatives_of_u
  ](auto tag) noexcept {
    using Tag = tmpl::type_from<decltype(tag)>;
    using DerivativeTag = Tags::deriv<Tag, tmpl::size_t<Dim>, DerivativeFrame>;
    auto& partial_derivatives_of_variable =
        get<DerivativeTag>(partial_derivatives_of_u);
    for (auto it = partial_derivatives_of_variable.begin();
         it != partial_derivatives_of_variable.end(); ++it) {
      const auto deriv_indices =
          partial_derivatives_of_variable.get_tensor_index(it);
      const size_t deriv_index = deriv_indices[0];
      const auto tensor_indices =
          all_but_specified_element_of<0>(deriv_indices);
      for (size_t d = 0; d < Dim; ++d) {
        *it += inverse_jacobian.get(d, deriv_index) *
               get<Tag>(gsl::at(logical_partial_derivatives_of_u, d))
                   .get(tensor_indices);
      }
    }
  });

  return partial_derivatives_of_u;
}

namespace partial_derivatives_detail {
template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<1, VariableTags, DerivativeTags> {
  static constexpr const size_t Dim = 1;
  static auto apply(const Variables<VariableTags>& u,
                    const Mesh<Dim>& mesh) noexcept {
    auto logical_partial_derivatives_of_u = make_array<Dim>(
        Variables<DerivativeTags>(u.number_of_grid_points(), 0.0));
    const Matrix& differentiation_matrix_xi =
        Spectral::differentiation_matrix(mesh.slice_through(0));
    dgemm_<true>('N', 'N', mesh.extents(0),
                 logical_partial_derivatives_of_u[0].size() / mesh.extents(0),
                 mesh.extents(0), 1.0, differentiation_matrix_xi.data(),
                 mesh.extents(0), u.data(), mesh.extents(0), 0.0,
                 logical_partial_derivatives_of_u[0].data(), mesh.extents(0));

    return logical_partial_derivatives_of_u;
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<2, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 2;
  static auto apply(const Variables<VariableTags>& u,
                    const Mesh<2>& mesh) noexcept {
    auto logical_partial_derivatives_of_u =
        make_array<Dim>(Variables<DerivativeTags>(u.number_of_grid_points()));
    const Matrix& differentiation_matrix_xi =
        Spectral::differentiation_matrix(mesh.slice_through(0));
    const size_t num_components_times_xi_slices =
        logical_partial_derivatives_of_u[0].size() / mesh.extents(0);
    dgemm_<true>('N', 'N', mesh.extents(0), num_components_times_xi_slices,
                 mesh.extents(0), 1.0, differentiation_matrix_xi.data(),
                 mesh.extents(0), u.data(), mesh.extents(0), 0.0,
                 logical_partial_derivatives_of_u[0].data(), mesh.extents(0));

    const auto u_eta_fastest =
        transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
            u, mesh.extents(0), num_components_times_xi_slices);
    Variables<DerivativeTags> partial_u_wrt_eta(u.number_of_grid_points());
    const Matrix& differentiation_matrix_eta =
        Spectral::differentiation_matrix(mesh.slice_through(1));
    const size_t num_components_times_eta_slices =
        logical_partial_derivatives_of_u[1].size() / mesh.extents(1);
    dgemm_<true>('N', 'N', mesh.extents(1), num_components_times_eta_slices,
                 mesh.extents(1), 1.0, differentiation_matrix_eta.data(),
                 mesh.extents(1), u_eta_fastest.data(), mesh.extents(1), 0.0,
                 partial_u_wrt_eta.data(), mesh.extents(1));
    transpose(make_not_null(&logical_partial_derivatives_of_u[1]),
              partial_u_wrt_eta, num_components_times_xi_slices,
              mesh.extents(0));

    return logical_partial_derivatives_of_u;
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<3, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 3;
  static auto apply(const Variables<VariableTags>& u,
                    const Mesh<3>& mesh) noexcept {
    auto logical_partial_derivatives_of_u =
        make_array<Dim>(Variables<DerivativeTags>(u.number_of_grid_points()));
    const Matrix& differentiation_matrix_xi =
        Spectral::differentiation_matrix(mesh.slice_through(0));
    const size_t num_components_times_xi_slices =
        logical_partial_derivatives_of_u[0].size() / mesh.extents(0);
    dgemm_<true>('N', 'N', mesh.extents(0), num_components_times_xi_slices,
                 mesh.extents(0), 1.0, differentiation_matrix_xi.data(),
                 mesh.extents(0), u.data(), mesh.extents(0), 0.0,
                 logical_partial_derivatives_of_u[0].data(), mesh.extents(0));

    auto u_eta_or_zeta_fastest =
        transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
            u, mesh.extents(0), num_components_times_xi_slices);
    Variables<DerivativeTags> partial_u_wrt_eta_or_zeta(
        u.number_of_grid_points());
    const Matrix& differentiation_matrix_eta =
        Spectral::differentiation_matrix(mesh.slice_through(1));
    const size_t num_components_times_eta_slices =
        logical_partial_derivatives_of_u[1].size() / mesh.extents(1);
    dgemm_<true>('N', 'N', mesh.extents(1), num_components_times_eta_slices,
                 mesh.extents(1), 1.0, differentiation_matrix_eta.data(),
                 mesh.extents(1), u_eta_or_zeta_fastest.data(), mesh.extents(1),
                 0.0, partial_u_wrt_eta_or_zeta.data(), mesh.extents(1));
    transpose(make_not_null(&logical_partial_derivatives_of_u[1]),
              partial_u_wrt_eta_or_zeta, num_components_times_xi_slices,
              mesh.extents(0));

    const size_t chunk_size = mesh.extents(0) * mesh.extents(1);
    const size_t number_of_chunks =
        logical_partial_derivatives_of_u[1].size() / chunk_size;
    transpose(make_not_null(&u_eta_or_zeta_fastest), u, chunk_size,
              number_of_chunks);
    const Matrix& differentiation_matrix_zeta =
        Spectral::differentiation_matrix(mesh.slice_through(2));
    const size_t num_components_times_zeta_slices =
        logical_partial_derivatives_of_u[2].size() / mesh.extents(2);
    dgemm_<true>('N', 'N', mesh.extents(2), num_components_times_zeta_slices,
                 mesh.extents(2), 1.0, differentiation_matrix_zeta.data(),
                 mesh.extents(2), u_eta_or_zeta_fastest.data(), mesh.extents(2),
                 0.0, partial_u_wrt_eta_or_zeta.data(), mesh.extents(2));
    transpose(make_not_null(&logical_partial_derivatives_of_u[2]),
              partial_u_wrt_eta_or_zeta, number_of_chunks, chunk_size);

    return logical_partial_derivatives_of_u;
  }
};
}  // namespace partial_derivatives_detail
