// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/ApplyTensorYlmFilter.hpp"

#include <cstddef>
#include <cstring>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.tpp"

namespace CurvedScalarWave {

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
    const SimpleSparseMatrix& filter_matrix_scalar,
    const SimpleSparseMatrix& filter_matrix_i, const size_t ell_max,
    const size_t radial_extents) {
  const auto& ylm = ylm::get_spherepack_cache(ell_max);
  ASSERT(
      radial_extents * ylm.physical_size() == sw_vars->number_of_grid_points(),
      "Mismatch " << radial_extents * ylm.physical_size() << " must equal "
                  << sw_vars->number_of_grid_points());
  if (temp_storage->number_of_grid_points() <=
      radial_extents * ylm.spectral_size()) {
    temp_storage->initialize(radial_extents * ylm.spectral_size());
  }

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
  ylm::TensorYlm::filter_detail::nodal_to_modal_ylm(
      make_not_null(&sw_spectral_vars), temp_grid_vars, ylm, radial_extents);

  // 3. Filter
  // src: sw_spectral_vars
  // dest: sw_spectral_vars
  // but using temp_grid_vars as temp storage for each tensor
  tmpl::for_each<filter_detail::sw_vars_list<Frame::Grid>>(
      [&sw_spectral_vars, &temp_grid_vars, radial_extents, &filter_matrix_i,
       &filter_matrix_scalar]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
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
            filter_matrix_i.increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          } else {
            filter_matrix_scalar.increment_multiply_on_right(
                make_not_null(&dest), offset, stride, src, offset, stride);
          }
        }
        // Copy the result for this tensor back into sw_spectral_vars.
        get<Tag>(sw_spectral_vars) = get<Tag>(dest_tensor);
      });

  // 4. Modal to nodal transformation.
  // src: sw_spectral_vars
  // dest: temp_grid_vars
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&temp_grid_vars), sw_spectral_vars, ylm, radial_extents);

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

TensorYlmFilter::TensorYlmFilter() = default;

TensorYlmFilter::TensorYlmFilter(CkMigrateMessage* msg)
    : Filters::Filter(msg) {}

TensorYlmFilter::TensorYlmFilter(const TensorYlmFilter& rhs)
    : Filters::Filter(rhs),
      num_modes_to_kill_(rhs.num_modes_to_kill_),
      half_power_(rhs.half_power_) {}

TensorYlmFilter& TensorYlmFilter::operator=(const TensorYlmFilter& rhs) {
  if (this != &rhs) {
    num_modes_to_kill_ = rhs.num_modes_to_kill_;
    half_power_ = rhs.half_power_;
  }
  return *this;
}

TensorYlmFilter::TensorYlmFilter(TensorYlmFilter&& rhs)
    : Filters::Filter(std::move(rhs)),
      num_modes_to_kill_(rhs.num_modes_to_kill_),
      half_power_(std::move(rhs.half_power_)) {}

TensorYlmFilter& TensorYlmFilter::operator=(TensorYlmFilter&& rhs) {
  if (this != &rhs) {
    num_modes_to_kill_ = rhs.num_modes_to_kill_;
    half_power_ = std::move(rhs.half_power_);
  }
  return *this;
}

TensorYlmFilter::TensorYlmFilter(const size_t num_modes_to_kill,
                                 std::optional<size_t> half_power)
    : num_modes_to_kill_(num_modes_to_kill), half_power_(half_power) {}

void TensorYlmFilter::pup(PUP::er& p) {
  Filters::Filter::pup(p);
  p | num_modes_to_kill_;
  p | half_power_;
  // The filter matrices and temp storage are lazily initialized,
  // so we don't pup them.
}

void TensorYlmFilter::operator()(
    const gsl::not_null<
        Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        sw_vars,
    const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial) const {
  if (mesh.basis(1) != Spectral::Basis::SphericalHarmonic) {
    return;
  }
  ASSERT(mesh.basis(2) == Spectral::Basis::SphericalHarmonic,
         "TensorYlmFilter requires spherical harmonic basis in both "
         "angular directions.");
  const size_t radial_extents = mesh.extents(0);
  const size_t l_max = mesh.extents(1) - 1;

  // Cache the filter matrices
  if (cached_l_max_ != l_max) {
    ylm::TensorYlm::fill_filter<Scalar<DataVector>::structure>(
        make_not_null(&filter_matrix_scalar_), l_max, num_modes_to_kill_,
        half_power_, normalization_);
    ylm::TensorYlm::fill_filter<tnsr::i<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_i_), l_max, num_modes_to_kill_,
        half_power_, normalization_);
    cached_l_max_ = l_max;
  }

  // Apply the filter
  const auto jac_inertial_to_grid =
      determinant_and_inverse(jac_grid_to_inertial).second;
  apply_tensor_ylm_filter(sw_vars, make_not_null(&temp_storage_),
                          jac_inertial_to_grid, jac_grid_to_inertial,
                          filter_matrix_scalar_, filter_matrix_i_, l_max,
                          radial_extents);
}

bool operator==(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs) {
  return lhs.num_modes_to_kill_ == rhs.num_modes_to_kill_ and
         lhs.half_power_ == rhs.half_power_;
}

bool operator!=(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID TensorYlmFilter::my_PUP_ID = 0;  // NOLINT

// Explicit instantiations

namespace filter_detail {
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
}  // namespace CurvedScalarWave

namespace ylm::TensorYlm::filter_detail {
YLM_TENSORYLM_INSTANTIATE_MODAL_NODAL_TRANSFORMS(
    CurvedScalarWave::filter_detail::sw_vars_list<Frame::Grid>);
YLM_TENSORYLM_INSTANTIATE_MODAL_NODAL_TRANSFORMS(
    CurvedScalarWave::filter_detail::sw_vars_list<Frame::Inertial>);
}  // namespace ylm::TensorYlm::filter_detail
