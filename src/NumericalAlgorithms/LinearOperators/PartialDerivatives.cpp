// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Transpose.hpp"
#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/MakeArray.hpp"

namespace partial_derivatives_detail {
template <size_t Dim, typename VariableTags, typename DerivativeTags>
struct LogicalImpl;
}  // namespace partial_derivatives_detail

template <typename DerivativeTags, typename VariableTags, size_t Dim>
std::array<Variables<DerivativeTags>, Dim> logical_partial_derivatives(
    const Variables<VariableTags>& u, const Index<Dim>& extents) noexcept {
  return partial_derivatives_detail::LogicalImpl<
      Dim, VariableTags, DerivativeTags>::apply(u, extents);
}

template <typename DerivativeTags, typename VariableTags, size_t Dim,
          typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::deriv, DerivativeTags, tmpl::size_t<Dim>,
                           DerivativeFrame>>
partial_derivatives(
    const Variables<VariableTags>& u, const Index<Dim>& extents,
    const Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                 typelist<SpatialIndex<Dim, UpLo::Up, Frame::Logical>,
                          SpatialIndex<Dim, UpLo::Lo, DerivativeFrame>>>&
        inverse_jacobian) noexcept {
  const auto logical_partial_derivatives_of_u =
      logical_partial_derivatives<DerivativeTags>(u, extents);

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
                    const Index<Dim>& extents) noexcept {
    auto logical_partial_derivatives_of_u = make_array<Dim>(
        Variables<DerivativeTags>(u.number_of_grid_points(), 0.0));
    const Matrix& differentiation_matrix_xi =
        Basis::lgl::differentiation_matrix(extents[0]);
    dgemm_('N', 'N', extents[0],
           logical_partial_derivatives_of_u[0].size() / extents[0], extents[0],
           1.0, differentiation_matrix_xi.data(), extents[0], u.data(),
           extents[0], 0.0, logical_partial_derivatives_of_u[0].data(),
           extents[0]);

    return logical_partial_derivatives_of_u;
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<2, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 2;
  static auto apply(const Variables<VariableTags>& u,
                    const Index<2>& extents) noexcept {
    auto logical_partial_derivatives_of_u =
        make_array<Dim>(Variables<DerivativeTags>(u.number_of_grid_points()));
    const Matrix& differentiation_matrix_xi =
        Basis::lgl::differentiation_matrix(extents[0]);
    const size_t num_components_times_xi_slices =
        logical_partial_derivatives_of_u[0].size() / extents[0];
    dgemm_('N', 'N', extents[0], num_components_times_xi_slices, extents[0],
           1.0, differentiation_matrix_xi.data(), extents[0], u.data(),
           extents[0], 0.0, logical_partial_derivatives_of_u[0].data(),
           extents[0]);

    const auto u_eta_fastest =
        transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
            u, extents[0], num_components_times_xi_slices);
    Variables<DerivativeTags> partial_u_wrt_eta(u.number_of_grid_points());
    const Matrix& differentiation_matrix_eta =
        Basis::lgl::differentiation_matrix(extents[1]);
    const size_t num_components_times_eta_slices =
        logical_partial_derivatives_of_u[1].size() / extents[1];
    dgemm_('N', 'N', extents[1], num_components_times_eta_slices, extents[1],
           1.0, differentiation_matrix_eta.data(), extents[1],
           u_eta_fastest.data(), extents[1], 0.0, partial_u_wrt_eta.data(),
           extents[1]);
    transpose(partial_u_wrt_eta, num_components_times_xi_slices, extents[0],
              make_not_null(&logical_partial_derivatives_of_u[1]));

    return logical_partial_derivatives_of_u;
  }
};

template <typename VariableTags, typename DerivativeTags>
struct LogicalImpl<3, VariableTags, DerivativeTags> {
  static constexpr size_t Dim = 3;
  static auto apply(const Variables<VariableTags>& u,
                    const Index<3>& extents) noexcept {
    auto logical_partial_derivatives_of_u =
        make_array<Dim>(Variables<DerivativeTags>(u.number_of_grid_points()));
    const Matrix& differentiation_matrix_xi =
        Basis::lgl::differentiation_matrix(extents[0]);
    const size_t num_components_times_xi_slices =
        logical_partial_derivatives_of_u[0].size() / extents[0];
    dgemm_('N', 'N', extents[0], num_components_times_xi_slices, extents[0],
           1.0, differentiation_matrix_xi.data(), extents[0], u.data(),
           extents[0], 0.0, logical_partial_derivatives_of_u[0].data(),
           extents[0]);

    auto u_eta_or_zeta_fastest =
        transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
            u, extents[0], num_components_times_xi_slices);
    Variables<DerivativeTags> partial_u_wrt_eta_or_zeta(
        u.number_of_grid_points());
    const Matrix& differentiation_matrix_eta =
        Basis::lgl::differentiation_matrix(extents[1]);
    const size_t num_components_times_eta_slices =
        logical_partial_derivatives_of_u[1].size() / extents[1];
    dgemm_('N', 'N', extents[1], num_components_times_eta_slices, extents[1],
           1.0, differentiation_matrix_eta.data(), extents[1],
           u_eta_or_zeta_fastest.data(), extents[1], 0.0,
           partial_u_wrt_eta_or_zeta.data(), extents[1]);
    transpose(partial_u_wrt_eta_or_zeta, num_components_times_xi_slices,
              extents[0], make_not_null(&logical_partial_derivatives_of_u[1]));

    const size_t chunk_size = extents[0] * extents[1];
    const size_t number_of_chunks =
        logical_partial_derivatives_of_u[1].size() / chunk_size;
    transpose<Variables<VariableTags>, Variables<DerivativeTags>>(
        u, chunk_size, number_of_chunks, make_not_null(&u_eta_or_zeta_fastest));
    const Matrix& differentiation_matrix_zeta =
        Basis::lgl::differentiation_matrix(extents[2]);
    const size_t num_components_times_zeta_slices =
        logical_partial_derivatives_of_u[2].size() / extents[2];
    dgemm_('N', 'N', extents[2], num_components_times_zeta_slices, extents[2],
           1.0, differentiation_matrix_zeta.data(), extents[2],
           u_eta_or_zeta_fastest.data(), extents[2], 0.0,
           partial_u_wrt_eta_or_zeta.data(), extents[2]);
    transpose(partial_u_wrt_eta_or_zeta, number_of_chunks, chunk_size,
              make_not_null(&logical_partial_derivatives_of_u[2]));

    return logical_partial_derivatives_of_u;
  }
};
}  // namespace partial_derivatives_detail
