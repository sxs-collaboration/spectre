// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/ApplyTensorYlmFilter.hpp"

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
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    const gsl::not_null<Variables<sw_vars_list<DestFrame>>*> dest,
    const Variables<sw_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac) {
  const auto& [src_psi, src_pi, src_phi] = src;
  auto& [dest_psi, dest_pi, dest_phi] = *dest;

  // Just copy the scalars.
  get<>(dest_psi) = get<>(src_psi);
  get<>(dest_pi) = get<>(src_pi);

  // Now do the vector.
  for (size_t i = 0; i < 3; ++i) {
    dest_phi.get(i) = jac.get(0, i) * src_phi.get(0) +
                      jac.get(1, i) * src_phi.get(1) +
                      jac.get(2, i) * src_phi.get(2);
  }
}
}  // namespace filter_detail

template <>
void fill_tensor_ylm_filters<filter_detail::sw_vars_list<Frame::Inertial>>(
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

  matrix->number_of_ell_modes_to_kill = number_of_ell_modes_to_kill;
  matrix->half_power = half_power;
  matrix->coefficient_normalization = coefficient_normalization;
}

template <>
void apply_tensor_ylm_filter(
    const gsl::not_null<
        Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        sw_vars,
    const gsl::not_null<
        Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, const size_t ell_max,
    const size_t radial_extents) {
  const auto& ylm = ylm::get_spherepack_cache(ell_max);
  ASSERT(
      radial_extents * ylm.physical_size() == sw_vars->number_of_grid_points(),
      "Mismatch " << radial_extents * ylm.physical_size() << " must equal "
                  << sw_vars->number_of_grid_points());
  ASSERT(radial_extents * ylm.spectral_size() <=
             temp_storage->number_of_grid_points(),
         "Mismatch " << radial_extents * ylm.spectral_size() << " must be <= "
                     << temp_storage->number_of_grid_points());

  // Here we re-use the same memory multiple times.  Note that
  // 1. sw_vars_to_filter has the same number of components as
  //    sw_spatial_decomp_vars, even though the components are arranged
  //    differently. So we can create a non-owning Variables of either
  //    tag that points into the storage of a Variables with the opposite tag.
  // 2. temp_storage has a larger size than sw_vars, because temp_storage
  //    is sized to hold spectral coefficients (in S2) and sw_vars holds
  //    collocation points (in S2).  This means that we can create a
  //    non-owning Variables to hold collocation points but that points into
  //    temp_storage (but we cannot create a non-owning Variables to hold
  //    spectral coefficients that points into sw_vars).
  //
  // We define two different Variables that point into temp_storage
  // (and we should not use any them simultaneously) and one
  // Variables that points into sw_vars (which we should not use
  // simultaneously with sw_vars).
  Variables<filter_detail::sw_vars_list<Frame::Grid>> sw_spectral_vars(
      temp_storage->data(), temp_storage->size());
  Variables<filter_detail::sw_vars_list<Frame::Grid>> temp_grid_vars(
      sw_vars->data(), sw_vars->size());
  // The following Variables uses sw_vars->size() which is smaller
  // than temp_storage->size().
  ASSERT(sw_vars->size() <= temp_storage->size(),
         "Should have " << sw_vars->size() << " <= " << temp_storage->size());
  Variables<filter_detail::sw_vars_list<Frame::Grid>> temp_sw_vars(
      temp_storage->data(), sw_vars->size());

  // 1. Multiply by inverse Jacobians to get into (mostly) grid frame.
  //    It's not really the grid frame because there are no Hessian
  //    corrections, but those don't matter for this purpose.
  // src: sw_vars
  // dest: temp_sw_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Inertial, Frame::Grid>(make_not_null(&temp_sw_vars), *sw_vars,
                                    jac_inertial_to_grid);

  // 1a. Copy
  // src: temp_sw_vars
  // dest: temp_grid_vars
  std::memcpy(temp_grid_vars.data(), temp_sw_vars.data(),
              temp_sw_vars.size() * sizeof(double));

  // 2. Nodal to modal transformation.
  // src: temp_grid_vars
  // dest: sw_spectral_vars
  filter_detail::nodal_to_modal_ylm(make_not_null(&sw_spectral_vars),
                                    temp_grid_vars, ylm, radial_extents);

  // 3. Filter
  // src: sw_spectral_vars
  // dest: sw_spectral_vars
  // but using temp_grid_vars as temp storage for each tensor
  tmpl::for_each<filter_detail::sw_vars_list<Frame::Grid>>(
      [&sw_spectral_vars, &temp_grid_vars, radial_extents,
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
        // temp_grid_vars.  temp_grid_vars is larger than any
        // *SINGLE* tensor in sw_spectral_vars, so this is ok.
        // Note that sw_spectral_vars.number_of_grid_points()
        // is used for the size because that is the spectral size.
        ASSERT(sw_spectral_vars.number_of_grid_points() *
                       num_independent_components <=
                   temp_grid_vars.size(),
               "Insufficient size: must have "
                   << sw_spectral_vars.number_of_grid_points() *
                          num_independent_components
                   << " <= " << temp_grid_vars.size());

        Variables<tmpl::list<Tag>> dest_tensor(
            temp_grid_vars.data(), sw_spectral_vars.number_of_grid_points() *
                                       num_independent_components);

        // Delta term
        get<Tag>(dest_tensor) = get<Tag>(sw_spectral_vars);
        // The rest of the terms.

        // Here we assume that different components in a given
        // tensor are stored contiguously in memory, so we can grab a
        // pointer to the first component of the tensor and pass that
        // pointer to increment_multiply_on_right.
        const gsl::span<double> src(
            get<Tag>(sw_spectral_vars)[0].data(),
            num_independent_components *
                sw_spectral_vars.number_of_grid_points());
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
          } else {
            ASSERT(
                filter_matrices.scalar.has_value(),
                "Filter matrix for 'scalar' not set in FilterMatrixHolder for "
                "TensorYlm filtering.");
            filter_matrices.scalar->increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          }
        }
        // Copy the result for this tensor back into sw_spectral_vars.
        get<Tag>(sw_spectral_vars) = get<Tag>(dest_tensor);
      });

  // 4. Modal to nodal transformation.
  // src: sw_spectral_vars
  // dest: temp_grid_vars
  filter_detail::modal_to_nodal_ylm(make_not_null(&temp_grid_vars),
                                    sw_spectral_vars, ylm, radial_extents);

  // 4a. Copy
  // src: temp_grid_vars
  // dest: temp_sw_vars
  std::memcpy(temp_sw_vars.data(), temp_grid_vars.data(),
              temp_grid_vars.size() * sizeof(double));

  // 5. Multiply by Jacobians to get back into inertial frame.
  // src: temp_sw_vars
  // dest: sw_vars
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians<
      Frame::Grid, Frame::Inertial>(sw_vars, temp_sw_vars,
                                    jac_grid_to_inertial);
}

// Explicit instantiations

namespace filter_detail {

template void nodal_to_modal_ylm<sw_vars_list<Frame::Grid>>(
    gsl::not_null<Variables<sw_vars_list<Frame::Grid>>*> modal,
    const Variables<sw_vars_list<Frame::Grid>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void nodal_to_modal_ylm<sw_vars_list<Frame::Inertial>>(
    gsl::not_null<Variables<sw_vars_list<Frame::Inertial>>*> modal,
    const Variables<sw_vars_list<Frame::Inertial>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void modal_to_nodal_ylm<sw_vars_list<Frame::Grid>>(
    gsl::not_null<Variables<sw_vars_list<Frame::Grid>>*> modal,
    const Variables<sw_vars_list<Frame::Grid>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void modal_to_nodal_ylm<sw_vars_list<Frame::Inertial>>(
    gsl::not_null<Variables<sw_vars_list<Frame::Inertial>>*> modal,
    const Variables<sw_vars_list<Frame::Inertial>>& nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Grid, Frame::Inertial>(
    gsl::not_null<Variables<sw_vars_list<Frame::Inertial>>*> dest,
    const Variables<sw_vars_list<Frame::Grid>>& src,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>& jac);

template void transform_spatial_tensors_to_different_frame_without_hessians<
    Frame::Inertial, Frame::Grid>(
    gsl::not_null<Variables<sw_vars_list<Frame::Grid>>*> dest,
    const Variables<sw_vars_list<Frame::Inertial>>& src,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>& jac);
}  // namespace filter_detail
}  // namespace ylm::TensorYlm
