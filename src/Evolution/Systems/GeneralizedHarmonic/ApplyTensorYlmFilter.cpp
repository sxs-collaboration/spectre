// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"

#include <cstddef>
#include <cstring>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/TensorYlm/Filter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/TensorYlm/ApplyFilter.tpp"

namespace ylm::TensorYlm {

namespace filter_detail {

void break_spacetime_vars_into_spatial_pieces(
    const gsl::not_null<Variables<gh_spatial_vars_list<Frame::Inertial>>*>
        spatial_vars,
    const Variables<gh_spacetime_vars_list>& spacetime_vars) {
  const auto& [metric, pi, phi] = spacetime_vars;
  auto& [g_00, g_0i, g_ij, pi_00, pi_0i, pi_ij, phi_k00, phi_ki0, phi_kij] =
      *spatial_vars;

  get<>(g_00) = get<0, 0>(metric);
  get<>(pi_00) = get<0, 0>(pi);
  for (size_t i = 0; i < 3; ++i) {
    g_0i.get(i) = metric.get(i + 1, 0);
    pi_0i.get(i) = pi.get(i + 1, 0);
    for (size_t j = i; j < 3; ++j) {
      g_ij.get(i, j) = metric.get(i + 1, j + 1);
      pi_ij.get(i, j) = pi.get(i + 1, j + 1);
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    phi_k00.get(k) = phi.get(k, 0, 0);
    for (size_t i = 0; i < 3; ++i) {
      phi_ki0.get(k, i) = phi.get(k, i + 1, 0);
      for (size_t j = i; j < 3; ++j) {
        phi_kij.get(k, i, j) = phi.get(k, i + 1, j + 1);
      }
    }
  }
}

void assemble_spacetime_vars_from_spatial_pieces(
    const gsl::not_null<Variables<gh_spacetime_vars_list>*> spacetime_vars,
    const Variables<gh_spatial_vars_list<Frame::Inertial>>& spatial_vars) {
  auto& [metric, pi, phi] = *spacetime_vars;
  const auto& [g_00, g_0i, g_ij, pi_00, pi_0i, pi_ij, phi_k00, phi_ki0,
               phi_kij] = spatial_vars;

  get<0, 0>(metric) = get<>(g_00);
  get<0, 0>(pi) = get<>(pi_00);
  for (size_t i = 0; i < 3; ++i) {
    metric.get(i + 1, 0) = g_0i.get(i);
    pi.get(i + 1, 0) = pi_0i.get(i);
    for (size_t j = i; j < 3; ++j) {
      metric.get(i + 1, j + 1) = g_ij.get(i, j);
      pi.get(i + 1, j + 1) = pi_ij.get(i, j);
    }
  }
  for (size_t k = 0; k < 3; ++k) {
    phi.get(k, 0, 0) = phi_k00.get(k);
    for (size_t i = 0; i < 3; ++i) {
      phi.get(k, i + 1, 0) = phi_ki0.get(k, i);
      for (size_t j = i; j < 3; ++j) {
        phi.get(k, i + 1, j + 1) = phi_kij.get(k, i, j);
      }
    }
  }
}

template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    const gsl::not_null<Variables<gh_spatial_vars_list<DestFrame>>*> dest,
    const Variables<gh_spatial_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac) {
  const auto& [src_g_00, src_g_0i, src_g_ij, src_pi_00, src_pi_0i, src_pi_ij,
               src_phi_k00, src_phi_ki0, src_phi_kij] = src;
  auto& [dest_g_00, dest_g_0i, dest_g_ij, dest_pi_00, dest_pi_0i, dest_pi_ij,
         dest_phi_k00, dest_phi_ki0, dest_phi_kij] = *dest;

  // Just copy the scalars.
  get<>(dest_g_00) = get<>(src_g_00);
  get<>(dest_pi_00) = get<>(src_pi_00);

  // Do phi_kij first, using other vars as temp storage.
  for (size_t k = 0; k < 3; ++k) {
    // First index, putting result in dest_g_ij.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        dest_g_ij.get(i, j) = jac.get(0, k) * src_phi_kij.get(0, i, j) +
                              jac.get(1, k) * src_phi_kij.get(1, i, j) +
                              jac.get(2, k) * src_phi_kij.get(2, i, j);
      }
    }
    // 2nd index, putting result in dest_phi_ki0.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        dest_phi_ki0.get(i, j) = jac.get(0, i) * dest_g_ij.get(0, j) +
                                 jac.get(1, i) * dest_g_ij.get(1, j) +
                                 jac.get(2, i) * dest_g_ij.get(2, j);
      }
    }
    // 3rd index, putting result in dest_phi_kij.
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        dest_phi_kij.get(k, i, j) = jac.get(0, j) * dest_phi_ki0.get(i, 0) +
                                    jac.get(1, j) * dest_phi_ki0.get(i, 1) +
                                    jac.get(2, j) * dest_phi_ki0.get(i, 2);
      }
    }
  }

  // Do phi_ki0, using other vars as temp storage.
  for (size_t k = 0; k < 3; ++k) {
    // 1st index, putting result in dest_g_0i
    for (size_t i = 0; i < 3; ++i) {
      dest_g_0i.get(i) = jac.get(0, k) * src_phi_ki0.get(0, i) +
                         jac.get(1, k) * src_phi_ki0.get(1, i) +
                         jac.get(2, k) * src_phi_ki0.get(2, i);
    }
    // 2nd index, putting result in dest_phi_ki0
    for (size_t i = 0; i < 3; ++i) {
      dest_phi_ki0.get(k, i) = jac.get(0, i) * dest_g_0i.get(0) +
                               jac.get(1, i) * dest_g_0i.get(1) +
                               jac.get(2, i) * dest_g_0i.get(2);
    }
  }

  // Do g_ij and pi_ij, using other vars as temp storage.
  for (size_t i = 0; i < 3; ++i) {
    // 1st index, putting result in dest_g_0i and dest_pi_0i
    for (size_t j = 0; j < 3; ++j) {
      dest_g_0i.get(j) = jac.get(0, i) * src_g_ij.get(0, j) +
                         jac.get(1, i) * src_g_ij.get(1, j) +
                         jac.get(2, i) * src_g_ij.get(2, j);
      dest_pi_0i.get(j) = jac.get(0, i) * src_pi_ij.get(0, j) +
                          jac.get(1, i) * src_pi_ij.get(1, j) +
                          jac.get(2, i) * src_pi_ij.get(2, j);
    }
    // 2nd index, putting result in g_ij and pi_ij
    for (size_t j = i; j < 3; ++j) {
      dest_g_ij.get(i, j) = jac.get(0, j) * dest_g_0i.get(0) +
                            jac.get(1, j) * dest_g_0i.get(1) +
                            jac.get(2, j) * dest_g_0i.get(2);
      dest_pi_ij.get(i, j) = jac.get(0, j) * dest_pi_0i.get(0) +
                             jac.get(1, j) * dest_pi_0i.get(1) +
                             jac.get(2, j) * dest_pi_0i.get(2);
    }
  }
  // Now do vectors
  for (size_t i = 0; i < 3; ++i) {
    dest_g_0i.get(i) = jac.get(0, i) * src_g_0i.get(0) +
                       jac.get(1, i) * src_g_0i.get(1) +
                       jac.get(2, i) * src_g_0i.get(2);
    dest_pi_0i.get(i) = jac.get(0, i) * src_pi_0i.get(0) +
                        jac.get(1, i) * src_pi_0i.get(1) +
                        jac.get(2, i) * src_pi_0i.get(2);
    dest_phi_k00.get(i) = jac.get(0, i) * src_phi_k00.get(0) +
                          jac.get(1, i) * src_phi_k00.get(1) +
                          jac.get(2, i) * src_phi_k00.get(2);
  }
}

}  // namespace filter_detail

template <>
void fill_tensor_ylm_filters<filter_detail::gh_spacetime_vars_list>(
    const gsl::not_null<FilterMatrixHolder*> matrix, const size_t ell_max,
    const size_t number_of_ell_modes_to_kill,
    const std::optional<size_t> half_power,
    const CoefficientNormalization coefficient_normalization) {
  const bool parameters_match =
      matrix->number_of_ell_modes_to_kill == number_of_ell_modes_to_kill and
      matrix->half_power == half_power and
      matrix->coefficient_normalization == coefficient_normalization;
  if (not parameters_match or not matrix->scalar.has_value()) {
    matrix->scalar = decltype(matrix->scalar)::value_type{};
    ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
        make_not_null(&matrix->scalar.value()), ell_max,
        number_of_ell_modes_to_kill, half_power, coefficient_normalization);
  }
  if (not parameters_match or not matrix->i.has_value()) {
    matrix->i = decltype(matrix->i)::value_type{};
    ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
        make_not_null(&matrix->i.value()), ell_max, number_of_ell_modes_to_kill,
        half_power, coefficient_normalization);
  }
  if (not parameters_match or not matrix->ii.has_value()) {
    matrix->ii = decltype(matrix->ii)::value_type{};
    ylm::TensorYlm::fill_filter<tnsr::ii<DataVector, 3>::structure>(
        make_not_null(&matrix->ii.value()), ell_max,
        number_of_ell_modes_to_kill, half_power, coefficient_normalization);
  }
  if (not parameters_match or not matrix->ij.has_value()) {
    matrix->ij = decltype(matrix->ij)::value_type{};
    ylm::TensorYlm::fill_filter<tnsr::ij<DataVector, 3>::structure>(
        make_not_null(&matrix->ij.value()), ell_max,
        number_of_ell_modes_to_kill, half_power, coefficient_normalization);
  }
  if (not parameters_match or not matrix->kii.has_value()) {
    matrix->kii = decltype(matrix->kii)::value_type{};
    ylm::TensorYlm::fill_filter<tnsr::ijj<DataVector, 3>::structure>(
        make_not_null(&matrix->kii.value()), ell_max,
        number_of_ell_modes_to_kill, half_power, coefficient_normalization);
  }

  matrix->number_of_ell_modes_to_kill = number_of_ell_modes_to_kill;
  matrix->half_power = half_power;
  matrix->coefficient_normalization = coefficient_normalization;
}

template <>
void apply_tensor_ylm_filter(
    const gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*>
        gh_vars,
    const gsl::not_null<Variables<filter_detail::gh_spacetime_vars_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const ylm::TensorYlm::FilterMatrixHolder& filter_matrices,
    const size_t ell_max, const size_t radial_extents) {
  const auto& ylm = ylm::get_spherepack_cache(ell_max);
  ASSERT(
      radial_extents * ylm.physical_size() == gh_vars->number_of_grid_points(),
      "Mismatch " << radial_extents * ylm.physical_size() << " must equal "
                  << gh_vars->number_of_grid_points());
  ASSERT(radial_extents * ylm.spectral_size() <=
             temp_storage->number_of_grid_points(),
         "Mismatch " << radial_extents * ylm.spectral_size() << " must be <= "
                     << temp_storage->number_of_grid_points());

  // Here we re-use the same memory multiple times.  Note that
  // 1. gh_vars_to_filter has the same number of components as
  //    gh_spatial_decomp_vars, even though the components are arranged
  //    differently. So we can create a non-owning Variables of either
  //    tag that points into the storage of a Variables with the opposite tag.
  // 2. temp_storage has a larger size than gh_vars, because temp_storage
  //    is sized to hold spectral coefficients (in S2) and gh_vars holds
  //    collocation points (in S2).  This means that we can create a
  //    non-owning Variables to hold collocation points but that points into
  //    temp_storage (but we cannot create a non-owning Variables to hold
  //    spectral coefficients that points into gh_vars).
  //
  // We define three different Variables that point into temp_storage
  // (and we should not use any two of them simultaneously) and one
  // Variables that points into gh_vars (which we should not use
  // simultaneously with gh_vars).
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      gh_spatial_spectral_vars(temp_storage->data(), temp_storage->size());
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> temp_spatial_vars(
      gh_vars->data(), gh_vars->size());
  // The following two Variables use gh_vars->size() which is smaller
  // than temp_storage->size().
  ASSERT(gh_vars->size() <= temp_storage->size(),
         "Should have " << gh_vars->size() << " <= " << temp_storage->size());
  const Variables<filter_detail::gh_spacetime_vars_list> temp_gh_vars(
      temp_storage->data(), gh_vars->size());
  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      gh_spatial_vars(temp_storage->data(), gh_vars->size());

  // 1. Break up into spatial pieces.
  // src: gh_vars
  // dest: gh_spatial_vars
  filter_detail::break_spacetime_vars_into_spatial_pieces(
      make_not_null(&gh_spatial_vars), *gh_vars);

  // 2. Multiply by inverse Jacobians to get into (mostly) grid frame.
  //    It's not really the grid frame because there are no Hessian
  //    corrections, but those don't matter for this purpose.
  // src: gh_spatial_vars
  // dest: temp_spatial_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&temp_spatial_vars),
                                    gh_spatial_vars, jac_inertial_to_grid);

  // 3. Nodal to modal transformation.
  // src: temp_spatial_vars
  // dest: gh_spatial_spectral_vars
  filter_detail::nodal_to_modal_ylm(make_not_null(&gh_spatial_spectral_vars),
                                    temp_spatial_vars, ylm, radial_extents);

  // 4. Filter
  // src: gh_spatial_spectral_vars
  // dest: gh_spatial_spectral_vars
  // but using temp_spatial_vars as temp storage for each tensor
  tmpl::for_each<filter_detail::gh_spatial_vars_list<Frame::Grid>>(
      [&gh_spatial_spectral_vars, &temp_spatial_vars, radial_extents,
       &filter_matrices]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        // Different compilers disagree on whether radial_extents
        // needs to be in the capture list of this lambda, and
        // whether radial_extents is 'used' in the lambda.
        // Adding it to the capture list and adding a cast here
        // satisfies everyone.
        (void)radial_extents;
        constexpr size_t num_independent_components =
            Tag::type::structure::size();
        // Create destination tensor: non-owning and pointing into
        // temp_spatial_vars.  temp_spatial_vars is larger than any
        // *SINGLE* tensor in gh_spatial_spectral_vars, so this is ok.
        // Note that gh_spatial_spectral_vars.number_of_grid_points()
        // is used for the size because that is the spectral size.
        ASSERT(gh_spatial_spectral_vars.number_of_grid_points() *
                       num_independent_components <=
                   temp_spatial_vars.size(),
               "Insufficient size: must have "
                   << gh_spatial_spectral_vars.number_of_grid_points() *
                          num_independent_components
                   << " <= " << temp_spatial_vars.size());

        Variables<tmpl::list<Tag>> dest_tensor(
            temp_spatial_vars.data(),
            gh_spatial_spectral_vars.number_of_grid_points() *
                num_independent_components);

        // Delta term
        get<Tag>(dest_tensor) = get<Tag>(gh_spatial_spectral_vars);
        // The rest of the terms.

        // Here we assume that different components in a given
        // tensor are stored contiguously in memory, so we can grab a
        // pointer to the first component of the tensor and pass that
        // pointer to increment_multiply_on_right.
        const gsl::span<double> src(
            get<Tag>(gh_spatial_spectral_vars)[0].data(),
            num_independent_components *
                gh_spatial_spectral_vars.number_of_grid_points());
        gsl::span<double> dest(
            get<Tag>(dest_tensor)[0].data(),
            num_independent_components * dest_tensor.number_of_grid_points());
        // If the mesh is 3-dimensional (i.e. radial_extents>1), then
        // we need to loop over offsets.  If not, then there's only
        // one loop iteration.
        const size_t stride = radial_extents;
        for (size_t offset = 0; offset < stride; ++offset) {
          // Each type of tensor gets a different filter matrix.
          if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                       Symmetry<1>>) {
            ASSERT(filter_matrices.i.has_value(),
                   "Filter matrix for 'i' not set in FilterMatrixHolder for "
                   "TensorYlm filtering.");
            filter_matrices.i->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else if constexpr (std::is_same_v<
                                   typename Tag::type::structure::symmetry,
                                   Symmetry<1, 1>>) {
            ASSERT(filter_matrices.ii.has_value(),
                   "Filter matrix for 'ii' not set in FilterMatrixHolder for "
                   "TensorYlm filtering.");
            filter_matrices.ii->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else if constexpr (std::is_same_v<
                                   typename Tag::type::structure::symmetry,
                                   Symmetry<2, 1>>) {
            ASSERT(filter_matrices.ij.has_value(),
                   "Filter matrix for 'ij' not set in FilterMatrixHolder for "
                   "TensorYlm filtering.");
            filter_matrices.ij->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else if constexpr (std::is_same_v<
                                   typename Tag::type::structure::symmetry,
                                   Symmetry<2, 1, 1>>) {
            ASSERT(filter_matrices.kii.has_value(),
                   "Filter matrix for 'kii' not set in FilterMatrixHolder for "
                   "TensorYlm filtering.");
            filter_matrices.kii->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else {
            ASSERT(
                filter_matrices.scalar.has_value(),
                "Filter matrix for 'scalar' not set in FilterMatrixHolder for "
                "TensorYlm filtering.");
            filter_matrices.scalar->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          }
        }
        // Copy the result for this tensor back into gh_spatial_spectral_vars.
        get<Tag>(gh_spatial_spectral_vars) = get<Tag>(dest_tensor);
      });

  // 5. Modal to nodal transformation.
  // src: gh_spatial_spectral_vars
  // dest: temp_spatial_vars
  filter_detail::modal_to_nodal_ylm(make_not_null(&temp_spatial_vars),
                                    gh_spatial_spectral_vars, ylm,
                                    radial_extents);

  // 6. Multiply by Jacobians to get back into inertial frame.
  // src: temp_spatial_vars
  // dest: gh_spatial_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Grid, Frame::Inertial>(make_not_null(&gh_spatial_vars),
                                    temp_spatial_vars, jac_grid_to_inertial);

  // 7. Put back into spacetime tensors.
  // src: gh_spatial_vars
  // dest: gh_vars
  filter_detail::assemble_spacetime_vars_from_spatial_pieces(gh_vars,
                                                             gh_spatial_vars);
}

// Explicit instantiations

namespace filter_detail {

template void nodal_to_modal_ylm<gh_spatial_vars_list<Frame::Grid>>(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Grid>>*> modal,
    const Variables<gh_spatial_vars_list<Frame::Grid>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void modal_to_nodal_ylm<gh_spatial_vars_list<Frame::Grid>>(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Grid>>*> modal,
    const Variables<gh_spatial_vars_list<Frame::Grid>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void nodal_to_modal_ylm<gh_spacetime_vars_list>(
    gsl::not_null<Variables<gh_spacetime_vars_list>*> modal,
    const Variables<gh_spacetime_vars_list>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void modal_to_nodal_ylm<gh_spacetime_vars_list>(
    gsl::not_null<Variables<gh_spacetime_vars_list>*> modal,
    const Variables<gh_spacetime_vars_list>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Grid, Frame::Inertial>(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Inertial>>*> dest,
    const Variables<gh_spatial_vars_list<Frame::Grid>>& src,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>& jac);

template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Inertial, Frame::Grid>(
    gsl::not_null<Variables<gh_spatial_vars_list<Frame::Grid>>*> dest,
    const Variables<gh_spatial_vars_list<Frame::Inertial>>& src,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>& jac);
}  // namespace filter_detail
}  // namespace ylm::TensorYlm
