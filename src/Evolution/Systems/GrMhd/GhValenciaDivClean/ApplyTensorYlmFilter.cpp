// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/GhValenciaDivClean/ApplyTensorYlmFilter.hpp"

#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/RadiationTransport/NoNeutrinos/System.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/TensorYlm/Filter.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

#include "NumericalAlgorithms/TensorYlm/ApplyFilter.tpp"

namespace ylm::TensorYlm {
template <>
void fill_tensor_ylm_filters<
    grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>(
    const gsl::not_null<FilterMatrixHolder*> matrix, const size_t ell_max,
    const size_t number_of_ell_modes_to_kill,
    const std::optional<size_t> half_power,
    const CoefficientNormalization coefficient_normalization) {
  fill_tensor_ylm_filters<filter_detail::gh_spacetime_vars_list>(
      matrix, ell_max, number_of_ell_modes_to_kill, half_power,
      coefficient_normalization);
}

template <>
void apply_tensor_ylm_filter(
    const gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        vars,
    const gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        temp_storage,
    const InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>&
        jac_inertial_to_grid,
    const InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>&
        jac_grid_to_inertial,
    const FilterMatrixHolder& filter_matrices, const size_t ell_max,
    const size_t radial_extents) {
  using gh_spacetime_vars_list = filter_detail::gh_spacetime_vars_list;
  using valencia_grid_vars_list =
      grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list;

  const auto& ylm = ylm::get_spherepack_cache(ell_max);
  // vars_num_pts is called N in comments for brevity
  const size_t vars_num_pts = vars->number_of_grid_points();
  // temp_num_pts is called M in comments for brevity
  const size_t temp_num_pts = temp_storage->number_of_grid_points();
  ASSERT(vars_num_pts == radial_extents * ylm.physical_size(),
         "Mismatch: vars has " << vars_num_pts << " grid points, expected "
                               << radial_extents * ylm.physical_size());
  ASSERT(temp_num_pts >= radial_extents * ylm.spectral_size(),
         "Mismatch: temp_storage has " << temp_num_pts
                                       << " grid points, need at least "
                                       << radial_extents * ylm.spectral_size());

  // ---- Step 1: Filter GH variables ----
  // The GH spacetime vars occupy the first gh_num_comps components of vars.
  // gh_spacetime_vars_list = [SpacetimeMetric(10), Pi(10), Phi(30)] = 50 comps
  constexpr size_t gh_num_comps =
      Variables<gh_spacetime_vars_list>::number_of_independent_components;

  // Non-owning Variables pointing to the GH portion of vars and temp_storage.
  Variables<gh_spacetime_vars_list> gh_vars(vars->data(),
                                            gh_num_comps * vars_num_pts);
  Variables<gh_spacetime_vars_list> gh_temp(temp_storage->data(),
                                            gh_num_comps * temp_num_pts);

  apply_tensor_ylm_filter(make_not_null(&gh_vars), make_not_null(&gh_temp),
                          jac_inertial_to_grid, jac_grid_to_inertial,
                          filter_matrices, ell_max, radial_extents);
  // The GH portion of vars is now filtered in place.

  // ---- Step 2: Filter Valencia variables ----
  // After the GH filter completes, temp_storage is free to reuse.
  //
  // Memory reuse plan (all pointers into temp_storage->data()):
  //   val_grid_vars:  [0, val_num_comps*N)        — Valencia physical, grid
  //   frame val_spec_vars:  [val_num_comps*N, ...)       — Valencia spectral,
  //   grid frame
  //
  // val_grid_vars is also used as scratch for dest_tensor during filtering.
  // Each dest_tensor for a single Valencia tag requires at most 3*M doubles,
  // and val_grid_vars holds val_num_comps*N >= 10*N > 3*M (since N > M).
  constexpr size_t val_num_comps =
      Variables<valencia_grid_vars_list>::number_of_independent_components;

  // Physical-space Valencia vars in grid frame (scratch)
  Variables<valencia_grid_vars_list> val_grid_vars(
      temp_storage->data(), val_num_comps * vars_num_pts);
  // Spectral Valencia vars in grid frame
  Variables<valencia_grid_vars_list> val_spec_vars(
      temp_storage->data() + val_num_comps * vars_num_pts,
      val_num_comps * temp_num_pts);

  // 2a. Transform Valencia vars from inertial to grid frame.
  //     Scalars: copy as-is.
  //     Covariant vector (TildeS): V_i^grid = jac_inertial_to_grid.get(j,i) *
  //     V_j^inertial Contravariant vector (TildeB): V^i_grid =
  //     jac_grid_to_inertial.get(i,j) * V^j_inertial
  get<grmhd::ValenciaDivClean::Tags::TildeD>(val_grid_vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeD>(*vars);
  get<grmhd::ValenciaDivClean::Tags::TildeYe>(val_grid_vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(*vars);
  get<grmhd::ValenciaDivClean::Tags::TildeTau>(val_grid_vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(*vars);
  get<grmhd::ValenciaDivClean::Tags::TildePhi>(val_grid_vars) =
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(*vars);

  const auto& tilde_s_inertial =
      get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(*vars);
  auto& tilde_s_grid =
      get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Grid>>(val_grid_vars);
  for (size_t i = 0; i < 3; ++i) {
    tilde_s_grid.get(i) =
        jac_inertial_to_grid.get(0, i) * tilde_s_inertial.get(0) +
        jac_inertial_to_grid.get(1, i) * tilde_s_inertial.get(1) +
        jac_inertial_to_grid.get(2, i) * tilde_s_inertial.get(2);
  }

  const auto& tilde_b_inertial =
      get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(*vars);
  auto& tilde_b_grid =
      get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Grid>>(val_grid_vars);
  for (size_t i = 0; i < 3; ++i) {
    tilde_b_grid.get(i) =
        jac_grid_to_inertial.get(i, 0) * tilde_b_inertial.get(0) +
        jac_grid_to_inertial.get(i, 1) * tilde_b_inertial.get(1) +
        jac_grid_to_inertial.get(i, 2) * tilde_b_inertial.get(2);
  }

  // 2b. Nodal to modal transform.
  filter_detail::nodal_to_modal_ylm(make_not_null(&val_spec_vars),
                                    val_grid_vars, ylm, radial_extents);

  // 2c. Apply filter matrix to each Valencia variable in spectral space.
  //     Scalars use filter_matrices.scalar; vectors use filter_matrices.i.
  tmpl::for_each<valencia_grid_vars_list>([&val_spec_vars, &val_grid_vars,
                                           radial_extents,
                                           &filter_matrices]<class Tag>(
                                              const tmpl::type_<Tag> /*meta*/) {
    (void)radial_extents;
    constexpr size_t num_independent_components = Tag::type::structure::size();
    // dest_tensor points into val_grid_vars (physical-space buffer, size N)
    // but is treated as having M = spectral_size grid points.
    ASSERT(val_spec_vars.number_of_grid_points() * num_independent_components <=
               val_grid_vars.size(),
           "Insufficient size: must have "
               << val_spec_vars.number_of_grid_points() *
                      num_independent_components
               << " <= " << val_grid_vars.size());
    Variables<tmpl::list<Tag>> dest_tensor(
        val_grid_vars.data(),
        val_spec_vars.number_of_grid_points() * num_independent_components);
    // Delta term
    get<Tag>(dest_tensor) = get<Tag>(val_spec_vars);
    // Filter matrix multiplication
    const gsl::span<double> src(
        get<Tag>(val_spec_vars)[0].data(),
        num_independent_components * val_spec_vars.number_of_grid_points());
    gsl::span<double> dest(
        get<Tag>(dest_tensor)[0].data(),
        num_independent_components * dest_tensor.number_of_grid_points());
    const size_t stride = radial_extents;
    for (size_t offset = 0; offset < stride; ++offset) {
      if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                   Symmetry<1>>) {
        // Vector
        filter_matrices.i->increment_multiply_on_right(
            make_not_null(&dest), offset, stride, src, offset, stride);
      } else {
        // Scalar
        filter_matrices.scalar->increment_multiply_on_right(
            make_not_null(&dest), offset, stride, src, offset, stride);
      }
    }
    get<Tag>(val_spec_vars) = get<Tag>(dest_tensor);
  });

  // 2d. Modal to nodal transform.
  filter_detail::modal_to_nodal_ylm(make_not_null(&val_grid_vars),
                                    val_spec_vars, ylm, radial_extents);

  // 2e. Transform back from grid to inertial frame.
  get<grmhd::ValenciaDivClean::Tags::TildeD>(*vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeD>(val_grid_vars);
  get<grmhd::ValenciaDivClean::Tags::TildeYe>(*vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeYe>(val_grid_vars);
  get<grmhd::ValenciaDivClean::Tags::TildeTau>(*vars) =
      get<grmhd::ValenciaDivClean::Tags::TildeTau>(val_grid_vars);
  get<grmhd::ValenciaDivClean::Tags::TildePhi>(*vars) =
      get<grmhd::ValenciaDivClean::Tags::TildePhi>(val_grid_vars);

  const auto& ts_grid =
      get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Grid>>(val_grid_vars);
  auto& ts_inertial =
      get<grmhd::ValenciaDivClean::Tags::TildeS<Frame::Inertial>>(*vars);
  for (size_t i = 0; i < 3; ++i) {
    ts_inertial.get(i) = jac_grid_to_inertial.get(0, i) * ts_grid.get(0) +
                         jac_grid_to_inertial.get(1, i) * ts_grid.get(1) +
                         jac_grid_to_inertial.get(2, i) * ts_grid.get(2);
  }

  const auto& tb_grid =
      get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Grid>>(val_grid_vars);
  auto& tb_inertial =
      get<grmhd::ValenciaDivClean::Tags::TildeB<Frame::Inertial>>(*vars);
  for (size_t i = 0; i < 3; ++i) {
    tb_inertial.get(i) = jac_inertial_to_grid.get(i, 0) * tb_grid.get(0) +
                         jac_inertial_to_grid.get(i, 1) * tb_grid.get(1) +
                         jac_inertial_to_grid.get(i, 2) * tb_grid.get(2);
  }
}

namespace filter_detail {

template void
nodal_to_modal_ylm<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>(
    gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        modal,
    const Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>&
        nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void
modal_to_nodal_ylm<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>(
    gsl::not_null<
        Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>*>
        nodal,
    const Variables<grmhd::GhValenciaDivClean::filter_detail::ghmhd_vars_list>&
        modal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void nodal_to_modal_ylm<
    grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>(
    gsl::not_null<Variables<
        grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>*>
        modal,
    const Variables<
        grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>&
        nodal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

template void modal_to_nodal_ylm<
    grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>(
    gsl::not_null<Variables<
        grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>*>
        nodal,
    const Variables<
        grmhd::GhValenciaDivClean::filter_detail::valencia_grid_vars_list>&
        modal,
    const ::ylm::Spherepack& ylm, size_t radial_extents);

}  // namespace filter_detail

}  // namespace ylm::TensorYlm
