// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/TensorYlmTransforms.hpp"

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.tpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm::TensorYlm {

namespace {

template <typename Tag>
void apply_cartesian_to_tensor_ylm_matrix(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        result,
    Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>&
        modal_coefficients,
    const SimpleSparseMatrix& matrix, const size_t radial_extents) {
  constexpr size_t num_independent_components = Tag::type::structure::size();
  ASSERT(result->number_of_grid_points() ==
             modal_coefficients.number_of_grid_points(),
         "Expected result and modal coefficients to have the same number of "
         "grid points, but got "
             << result->number_of_grid_points() << " and "
             << modal_coefficients.number_of_grid_points() << ".");

  // SimpleSparseMatrix::increment_multiply_on_right does `dest += matrix*src`,
  // so the destination must be zeroed for this pure basis transformation.
  for (auto& component : get<Tag>(*result)) {
    component = 0.0;
  }

  // Tensor components are stored contiguously, component by component. The
  // matrix acts on all components of one angular coefficient at a fixed radial
  // offset, so the radial index is handled as a strided matrix multiply.
  const gsl::span<double> src(
      get<Tag>(modal_coefficients)[0].data(),
      num_independent_components * modal_coefficients.number_of_grid_points());
  gsl::span<double> dest(
      get<Tag>(*result)[0].data(),
      num_independent_components * result->number_of_grid_points());
  for (size_t offset = 0; offset < radial_extents; ++offset) {
    matrix.increment_multiply_on_right(make_not_null(&dest), offset,
                                       radial_extents, src, offset,
                                       radial_extents);
  }
}

}  // namespace

void gh_variables_to_tensor_ylm_coefficients(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        gh_spatial_tensor_ylm_coefficients,
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        temp_storage,
    const Variables<filter_detail::gh_spacetime_vars_list>& gh_vars,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const SimpleSparseMatrix& cart_to_sphere_matrix_i,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ii,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ij,
    const SimpleSparseMatrix& cart_to_sphere_matrix_ijj,
    const Spherepack& spherepack, const size_t radial_extents) {
  // Check inputs' sizes are consistent. First check: transformations used here
  // assume l_max == m_max.
  ASSERT(spherepack.m_max() == spherepack.l_max(),
         "TensorYlm transforms require m_max == l_max because the "
         "cartesian-to-spherical matrices are built for the full set of m "
         "modes, but got l_max = "
             << spherepack.l_max() << " and m_max = " << spherepack.m_max()
             << ".");
  // As explained in the documentation for Spherepack::physical_size() and
  // Spherepack::spectral_size(), the spectral size is larger than the
  // physical size. Specifically, when l_max == m_max,
  // N_{\rm phys} = 2 L_{\rm max}^2 + 3 L_{\rm max} + 1,
  // N_{\rm spec} = 2 L_{\rm max}^2 + 4 L_{\rm max} + 2, so
  // N_{\rm spec} - N_{\rm phys} = L_{\rm max} + 1.
  // As a result, gh_spatial_tensor_ylm_coefficients and temp_storage
  // (which store spectral coefficients) have a larger size than gh_vars,
  // which stores the generalized harmonic variables at the collocation points.
  // Also note that spherepack.physical_size() and spherepack.spectral_size()
  // give the physical size and spectral sizes per radial collocation point.
  const size_t physical_size = radial_extents * spherepack.physical_size();
  const size_t spectral_size = radial_extents * spherepack.spectral_size();

  // gh_vars should have size physical_size, since it stores the generalized-
  // harmonic evolution variables at the collocation points. But the output,
  // gh_spatial_tensor_ylm_coefficients, must be spectral_size.
  ASSERT(gh_vars.number_of_grid_points() == physical_size,
         "Expected GH variables to have "
             << physical_size << " grid points, but got "
             << gh_vars.number_of_grid_points() << ".");
  ASSERT(gh_spatial_tensor_ylm_coefficients->number_of_grid_points() ==
             spectral_size,
         "Expected output tensor-Ylm coefficients to have "
             << spectral_size
             << " grid points, "
                "but got "
             << gh_spatial_tensor_ylm_coefficients->number_of_grid_points()
             << ".");

  // temp_storage must have enough room for all GH spatial variables, first at
  // the physical collocation points and then in spectral storage.
  ASSERT(temp_storage->number_of_grid_points() >= spectral_size,
         "Expected temporary storage to hold at least "
             << spectral_size
             << " grid points, but "
                "got "
             << temp_storage->number_of_grid_points() << ".");
  ASSERT(gh_spatial_tensor_ylm_coefficients->data() != temp_storage->data(),
         "The output tensor-Ylm coefficients must not alias temp_storage.");

  // The non-owning Variables below reinterpret the two caller-provided buffers
  // as either physical or spectral spatial GH variables. Only one view into a
  // given buffer should be used at a time.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      gh_spatial_inertial_vars(
          temp_storage->data(),
          physical_size *
              Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>::
                  number_of_independent_components);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_grid_vars(
          gh_spatial_tensor_ylm_coefficients->data(),
          physical_size *
              Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>::
                  number_of_independent_components);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_modal_vars(
          temp_storage->data(),
          spectral_size *
              Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>::
                  number_of_independent_components);

  // 1. Break into spatial pieces.
  // source:      gh_vars
  // destination: gh_spatial_inertial_vars, backed by temp_storage
  filter_detail::break_spacetime_vars_into_spatial_pieces(
      make_not_null(&gh_spatial_inertial_vars), gh_vars);

  // 2. Transform tensors to a "grid" frame. This transformation ignores
  //    Hessian corrections, so it isn't really a grid frame, but it's
  //    sufficient for getting the S2 modal coefficients.
  // source:      gh_spatial_inertial_vars, backed by temp_storage
  // destination: gh_spatial_grid_vars, backed by
  //              gh_spatial_tensor_ylm_coefficients
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&gh_spatial_grid_vars),
                                    gh_spatial_inertial_vars,
                                    jac_inertial_to_grid);

  // 3. Transform nodal values (values at collocation points) to modal values.
  // source:      gh_spatial_grid_vars, backed by
  //              gh_spatial_tensor_ylm_coefficients
  // destination: gh_spatial_modal_vars, backed by temp_storage
  filter_detail::nodal_to_modal_ylm(make_not_null(&gh_spatial_modal_vars),
                                    gh_spatial_grid_vars, spherepack,
                                    radial_extents);

  // 4. Transform from Cartesian to TensorYlm basis.
  //    Scalar values need no further correction, but tensors must be
  //    transformed from the Cartesian basis into the TensorYlm basis.
  // source:      gh_spatial_modal_vars, backed by temp_storage
  // destination: gh_spatial_tensor_ylm_coefficients
  tmpl::for_each<filter_detail::gh_spatial_vars_list<
      Frame::Grid>>([gh_spatial_tensor_ylm_coefficients, &gh_spatial_modal_vars,
                     radial_extents, &cart_to_sphere_matrix_i,
                     &cart_to_sphere_matrix_ii, &cart_to_sphere_matrix_ij,
                     &cart_to_sphere_matrix_ijj]<class Tag>(
                        const tmpl::type_<Tag> /*meta*/) {
    if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                 Symmetry<1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, gh_spatial_modal_vars,
          cart_to_sphere_matrix_i, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<1, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, gh_spatial_modal_vars,
          cart_to_sphere_matrix_ii, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, gh_spatial_modal_vars,
          cart_to_sphere_matrix_ij, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1, 1>>) {
      apply_cartesian_to_tensor_ylm_matrix<Tag>(
          gh_spatial_tensor_ylm_coefficients, gh_spatial_modal_vars,
          cart_to_sphere_matrix_ijj, radial_extents);
    } else {
      static_assert(Tag::type::rank() == 0,
                    "Unhandled tensor symmetry in TensorYlm transform.");
      get<Tag>(*gh_spatial_tensor_ylm_coefficients) =
          get<Tag>(gh_spatial_modal_vars);
    }
  });
}

}  // namespace ylm::TensorYlm
