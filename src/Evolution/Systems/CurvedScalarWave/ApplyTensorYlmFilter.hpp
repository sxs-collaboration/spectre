// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <optional>
#include <string>
#include <unordered_set>

#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlm.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

/// \cond
class DataVector;
namespace ylm {
class Spherepack;
}  // namespace ylm
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace CurvedScalarWave {

/// Defines tags and functions used internally in filtering, but
/// tested independently in the unit tests.
namespace filter_detail {

template <typename Frame>
using sw_vars_list = tmpl::list<Tags::Psi, Tags::Pi, Tags::Phi<3, Frame>>;

/*!
 * \brief Transforms spatial tensors into a different frame, ignoring hessians.
 *
 * This is done for filtering, where having the correct (i.e. with hessians)
 * transformation doesn't matter; all that matters is that the tensor
 * indices correspond to the coordinates (or in other words, no dual frame).
 *
 * Assumes that all the variables have lower indices.
 *
 * Takes special care to re-use memory. The Variables arguments must
 * already be allocated to their correct sizes; no memory allocation
 * is done.
 *
 * \tparam SrcFrame Source frame.
 * \tparam DestFrame Destination frame.
 * \param dest A Variables for the destination spatial variables.
 * \param src A Variables containing the source spatial variables.
 * \param jac The jacobian dx^src/dx^dest
 */
template <typename SrcFrame, typename DestFrame>
void transform_spatial_tensors_to_different_frame_without_hessians(
    gsl::not_null<Variables<sw_vars_list<DestFrame>>*> dest,
    const Variables<sw_vars_list<SrcFrame>>& src,
    const InverseJacobian<DataVector, 3, SrcFrame, DestFrame>& jac);

}  // namespace filter_detail

/*!
 * \brief Applies TensorYlm filter in place to Curved Scalar Wave variables.
 *
 * When radial_extents is 1, sw_vars and temp_storage are assumed to
 * be defined on a spherical slice, with number of grid points
 * corresponding to a spherical-harmonic grid of ell_max, and the
 * filter happens only on that slice.
 *
 * When radial_extents is > 1, sw_vars and temp_storage are assumed to
 * be defined on a spherical shell of topology I1 x S2. The filter
 * happens in the entire volume, internally iterating over each
 * spherical slice at a time.
 *
 * For performance reasons, apply_tensor_ylm_filter does not allocate
 * or deallocate memory, but it does take a temp_storage buffer.  The
 * size of temp_storage should at least
 * radial_extents*spectral_size*num_components, where num_components
 * is the total number of independent components in the SW variable
 * list (i.e. 5), and spectral_size is the size of the S2 Spherepack
 * spectral coefficient array for ell_max, as obtained from the member
 * function ylm::Spherepack::spectral_size().  Note that for S2 on
 * Spherepack, the number of collocation points is different than the
 * number of spectral coefficients, and both are different than the
 * size of the Spherepack storage array.
 *
 * \param sw_vars Scalar wave variables at collocation points.
 * \param temp_storage Temporary storage for scalar wave variables,
 *   allocated outside apply_tensor_ylm_filter. See above for size requirements.
 * \param jac_inertial_to_grid Jacobian taking V_x from inertial to grid.
 * \param jac_grid_to_inertial Jacobian taking V_x from grid to inertial.
 * \param filter_matrix_scalar The scalar filter matrix computed by fill_filter.
 * \param filter_matrix_i The Rank-1 matrix computed by fill_filter.
 * \param ell_max The maximum ylm ell.
 * \param radial_extents The number of radial grid points, can be 1 for slices.
 */
void apply_tensor_ylm_filter(
    gsl::not_null<Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        sw_vars,
    gsl::not_null<Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const SimpleSparseMatrix& filter_matrix_scalar,
    const SimpleSparseMatrix& filter_matrix_i, size_t ell_max,
    size_t radial_extents);

/*!
 * \brief DataBox mutator that applies a TensorYlm filter to the Curved Scalar
 * Wave variables and caches the filter matrices.
 */
class TensorYlmFilter : public Filters::Filter {
 public:
  struct NumModesToKill {
    using type = size_t;
    static constexpr Options::String help =
        "How many of the top ell modes to set to zero";
  };
  struct HalfPower {
    using type = Options::Auto<size_t, Options::AutoLabel::None>;
    static constexpr Options::String help =
        "The half-power sigma for more complicated filtering. "
        "If None, implements a Heaviside filter.";
  };
  using options = tmpl::list<NumModesToKill, HalfPower>;
  static constexpr Options::String help = {"Tensor Ylm filter."};

  TensorYlmFilter();
  TensorYlmFilter(const TensorYlmFilter& rhs);
  TensorYlmFilter& operator=(const TensorYlmFilter& rhs);
  TensorYlmFilter(TensorYlmFilter&& rhs);
  TensorYlmFilter& operator=(TensorYlmFilter&& rhs);
  ~TensorYlmFilter() override = default;

  WRAPPED_PUPable_decl_template(TensorYlmFilter);  // NOLINT
  explicit TensorYlmFilter(CkMigrateMessage* msg);

  TensorYlmFilter(size_t num_modes_to_kill, std::optional<size_t> half_power);

  std::optional<std::unordered_set<std::string>> blocks_to_filter()
      const override {
    return std::nullopt;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 public:  // DataBox-mutator protocol
  using argument_tags = tmpl::list<
      domain::Tags::Mesh<3>,
      domain::Tags::InverseJacobian<3, Frame::Grid, Frame::Inertial>>;

  void operator()(
      gsl::not_null<Variables<filter_detail::sw_vars_list<Frame::Inertial>>*>
          sw_vars,
      const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
          jac_grid_to_inertial) const;

 private:
  friend bool operator==(const TensorYlmFilter& lhs,
                         const TensorYlmFilter& rhs);

  size_t num_modes_to_kill_{0};
  std::optional<size_t> half_power_{std::nullopt};
  // Use Spherepack normalization because the variables are stored as Spherepack
  // modes
  static constexpr ylm::TensorYlm::CoefficientNormalization normalization_ =
      ylm::TensorYlm::CoefficientNormalization::Spherepack;
  // Caches and memory buffers
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t cached_l_max_{0};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SimpleSparseMatrix filter_matrix_scalar_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable SimpleSparseMatrix filter_matrix_i_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable Variables<filter_detail::sw_vars_list<Frame::Inertial>>
      temp_storage_{};
};

bool operator!=(const TensorYlmFilter& lhs, const TensorYlmFilter& rhs);

}  // namespace CurvedScalarWave
