// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm::TensorYlm {
namespace {
void test_break_spacetime_vars_into_spatial_pieces() {
  constexpr size_t mesh_size = 10;

  Variables<filter_detail::gh_spacetime_vars_list> gh_spacetime_vars(mesh_size);

  // Fill with random numbers
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < gh_spacetime_vars.size(); ++i) {
    gh_spacetime_vars.data()[i] = dist(generator);
  }

  // Break into spatial pieces, then do the inverse, and make
  // sure we get the original back.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      gh_spatial_vars(mesh_size);
  filter_detail::break_spacetime_vars_into_spatial_pieces(
      make_not_null(&gh_spatial_vars), gh_spacetime_vars);
  Variables<filter_detail::gh_spacetime_vars_list> test_gh_spacetime_vars(
      mesh_size);
  filter_detail::assemble_spacetime_vars_from_spatial_pieces(
      make_not_null(&test_gh_spacetime_vars), gh_spatial_vars);

  // This should be equal to the last bit, since we aren't doing
  // any operations that should incur roundoff error.
  for (size_t i = 0; i < gh_spacetime_vars.size(); ++i) {
    CHECK(gh_spacetime_vars.data()[i] == test_gh_spacetime_vars.data()[i]);
  }
}

void test_transform_spatial_tensors_to_different_frame() {
  constexpr size_t mesh_size = 10;

  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>> inertial_vars(
      mesh_size);

  // Fill inertial_vars with random numbers
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    inertial_vars.data()[i] = dist(generator);
  }

  // Create a Jacobian that is filled with random numbers, but make
  // sure that the jacobian is invertible.  Do this by making the
  // Jacobian diagonally dominant.
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jac(mesh_size);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac.get(i, j) = 0.05 * dist(generator);
    }
    jac.get(i, i) += positive_dist(generator);
  }

  // Invert the Jacobian
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial> invjac(
      mesh_size);
  Scalar<DataVector> det(mesh_size);
  determinant_and_inverse(make_not_null(&det), make_not_null(&invjac), jac);

  // Transform and inverse-transform, and make sure we get what
  // we started with.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> grid_vars(
      mesh_size);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians(
      make_not_null(&grid_vars), inertial_vars, jac);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      test_inertial_vars(mesh_size);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians(
      make_not_null(&test_inertial_vars), grid_vars, invjac);

  // We may get roundoff differences because there is multiplication
  // by Jacobians.
  for (size_t i = 0; i < inertial_vars.size(); ++i) {
    CHECK(inertial_vars.data()[i] == approx(test_inertial_vars.data()[i]));
  }
}

void test_modal_nodal_invertibility() {
  constexpr size_t radial_extents = 3;
  constexpr size_t ell_max = 4;

  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t spectral_mesh_size = ylm.spectral_size() * radial_extents;
  const size_t physical_mesh_size = ylm.physical_size() * radial_extents;

  // Fill modal variables with random numbers in each mode.
  // Note that we fill only valid modes in the storage.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> modal_vars(
      spectral_mesh_size);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<filter_detail::gh_spatial_vars_list<Frame::Grid>>(
      [&modal_vars, &it, &dist,
       &generator]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        auto& tensor = get<Tag>(modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              component[it() + offset] = dist(generator);
            }
          }
        }
      });

  // Do modal to nodal, and inverse transformation.
  // We do modal->nodal->modal rather than nodal->modal->nodal because
  // for S2, nodal points have more degrees of freedom than nodal values,
  // i.e. arbitrary nodal values are not necessarily representable as modes.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> nodal_vars(
      physical_mesh_size);
  filter_detail::modal_to_nodal_ylm(make_not_null(&nodal_vars), modal_vars, ylm,
                                    radial_extents);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> test_modal_vars(
      spectral_mesh_size, 0.0);
  filter_detail::nodal_to_modal_ylm(make_not_null(&test_modal_vars), nodal_vars,
                                    ylm, radial_extents);

  // We should get (modulo roundoff) what we started with.
  tmpl::for_each<filter_detail::gh_spatial_vars_list<Frame::Grid>>(
      [&modal_vars, &test_modal_vars,
       &it]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        constexpr size_t num_independent_components =
            Tag::type::structure::size();
        const auto& tensor_a = get<Tag>(modal_vars);
        const auto& tensor_b = get<Tag>(test_modal_vars);
        for (size_t storage_index = 0;
             storage_index < num_independent_components; ++storage_index) {
          const auto& a = tensor_a[storage_index];
          const auto& b = tensor_b[storage_index];
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              CHECK(a[it() + offset] == approx(b[it() + offset]));
            }
          }
        }
      });
}

template<typename VarsList>
void test_apply_filter(const size_t num_to_kill) {
  constexpr size_t radial_extents = 2;
  constexpr size_t ell_max = 9;

  const auto& ylm = ::ylm::get_spherepack_cache(ell_max);
  const size_t spectral_mesh_size = ylm.spectral_size() * radial_extents;
  const size_t physical_mesh_size = ylm.physical_size() * radial_extents;

  // Fill modal variables with random numbers in each mode.
  // Note that we fill only valid modes in the storage.
  Variables<VarsList> inertial_modal_vars(
      spectral_mesh_size, 0.0);
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  ylm::SpherepackIterator it(ell_max, ell_max, radial_extents, true);
  tmpl::for_each<VarsList>(
      [&inertial_modal_vars, &it, &dist,
       &generator]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        auto& tensor = get<Tag>(inertial_modal_vars);
        for (auto& component : tensor) {
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              // Do not set modes with ell > ell_max - rank
              // because those modes are incompletely represented by
              // the TensorYlm basis.
              if (it.l() <= ell_max - tensor.rank()) {
                component[it() + offset] = dist(generator);
              }
            }
          }
        }
      });

  // Do modal to nodal.
  Variables<VarsList> inertial_nodal_vars(
      physical_mesh_size);
  filter_detail::modal_to_nodal_ylm(make_not_null(&inertial_nodal_vars),
                                    inertial_modal_vars, ylm, radial_extents);

  // Save a copy of the modal vars
  const Variables<VarsList>
      test_inertial_modal_vars(inertial_modal_vars);

  // Even if num_to_kill is zero, the filter does
  // scalarylm->tensorylm->scalarylm without any additional cutting
  // off of modes.  This isn't really doing nothing, because any modes
  // in the scalarylm basis that are incompletely represented by the
  // tensorylm basis will be modified, but it is the best we can do.
  SimpleSparseMatrix filter_matrix_scalar;
  SimpleSparseMatrix filter_matrix_i;
  SimpleSparseMatrix filter_matrix_ii;
  SimpleSparseMatrix filter_matrix_ij;
  SimpleSparseMatrix filter_matrix_kii;
  fill_filter<Scalar<DataVector>::structure>(
      make_not_null(&filter_matrix_scalar), ell_max, num_to_kill, std::nullopt);
  fill_filter<tnsr::i<DataVector, 3>::structure>(
      make_not_null(&filter_matrix_i), ell_max, num_to_kill, std::nullopt);
  if constexpr (std::is_same_v<
                    VarsList, typename filter_detail::gh_spacetime_vars_list>) {
    fill_filter<tnsr::ii<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_ii), ell_max, num_to_kill, std::nullopt);
    fill_filter<tnsr::ij<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_ij), ell_max, num_to_kill, std::nullopt);
    fill_filter<tnsr::ijj<DataVector, 3>::structure>(
        make_not_null(&filter_matrix_kii), ell_max, num_to_kill, std::nullopt);
  }

  // Make up bogus jacobians with random numbers, and make
  // them diagonally dominant so that they are invertible.
  std::uniform_real_distribution<double> positive_dist{0.5, 1.0};
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid>
      jac_inertial_to_grid(physical_mesh_size);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      jac_inertial_to_grid.get(i, j) = 0.05 * dist(generator);
    }
    jac_inertial_to_grid.get(i, i) += positive_dist(generator);
  }

  // Invert the Jacobian
  InverseJacobian<DataVector, 3, Frame::Grid, Frame::Inertial>
      jac_grid_to_inertial(physical_mesh_size);
  Scalar<DataVector> det(physical_mesh_size);
  determinant_and_inverse(make_not_null(&det),
                          make_not_null(&jac_grid_to_inertial),
                          jac_inertial_to_grid);

  // Now carry out the entire filtering algorithm,
  // using inertial_modal_vars as temporary storage.
  if constexpr (std::is_same_v<
                    VarsList, typename filter_detail::gh_spacetime_vars_list>) {
    apply_tensor_ylm_filter(make_not_null(&inertial_nodal_vars),
                            make_not_null(&inertial_modal_vars),
                            jac_inertial_to_grid, jac_grid_to_inertial,
                            filter_matrix_scalar, filter_matrix_i,
                            filter_matrix_ii, filter_matrix_ij,
                            filter_matrix_kii, ell_max, radial_extents);
  } else {
    apply_tensor_ylm_filter(make_not_null(&inertial_nodal_vars),
                            make_not_null(&inertial_modal_vars),
                            jac_inertial_to_grid, jac_grid_to_inertial,
                            filter_matrix_scalar, filter_matrix_i, ell_max,
                            radial_extents);
  }

  // Do nodal to modal of the result (for comparison).
  filter_detail::nodal_to_modal_ylm(make_not_null(&inertial_modal_vars),
                                    inertial_nodal_vars, ylm, radial_extents);

  // We should get back the original modal vars, to roundoff.
  tmpl::for_each<VarsList>(
      [&inertial_modal_vars, &test_inertial_modal_vars, &ell_max, &num_to_kill,
       &it]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        constexpr size_t num_independent_components =
            Tag::type::structure::size();
        const auto& tensor_a = get<Tag>(inertial_modal_vars);
        const auto& tensor_b = get<Tag>(test_inertial_modal_vars);
        for (size_t storage_index = 0;
             storage_index < num_independent_components; ++storage_index) {
          const auto& a = tensor_a[storage_index];
          const auto& b = tensor_b[storage_index];
          for (size_t offset = 0; offset < radial_extents; ++offset) {
            for (it.reset(); it; ++it) {
              // If num_to_kill is zero, then all the coefs
              // should agree with the originals.
              // If num_to_kill is nonzero, then:
              //  - lcut is the largest mode that is LEFT ALONE
              //    in the Spin-weighted basis.  So lcut=lmax-num_to_kill
              //  - In the Cartesian basis, lcut+rank+1 is the smallest
              //    mode that is zeroed.  This is lmax-num_to_kill+rank+1.
              //  - In the Cartesian basis, lcut-rank is the largest mode
              //    that is unaffected. This is lmax-num_to_kill-rank.
              // Therefore all coefs
              // with (ell <= ell_max - num_to_kill - rank) should agree
              // with the originals because they have not been affected, and
              // all the modes with (ell >= ell_max - num_to_kill + rank+1)
              // should be zero because they have been killed by the filter.
              // For modes between those cases, they are modified in some
              // complicated way that we do not check here.
              if (num_to_kill == 0 or
                  it.l() <= ell_max - num_to_kill - tensor_b.rank()) {
                CAPTURE(ell_max);
                CAPTURE(it.l());
                CAPTURE(num_to_kill);
                CAPTURE(tensor_b.rank());
                CHECK(a[it() + offset] == approx(b[it() + offset]));
              } else if (it.l() >=
                         ell_max - num_to_kill + tensor_a.rank() + 1) {
                // The two tensors have the same rank, so it doesn't matter
                // whether we use tensor_a.rank() or tensor_b.rank() here.
                CAPTURE(ell_max);
                CAPTURE(it.l());
                CAPTURE(num_to_kill);
                CAPTURE(tensor_a.rank());
                CHECK(0.0 == approx(a[it() + offset]));
              }
            }
          }
        }
      });
}

// Debug builds are timing out slightly, so increase the timeout.
// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.SphericalHarmonics.ApplyTensorYlmFilter",
                  "[NumericalAlgorithms][Unit]") {
  test_break_spacetime_vars_into_spatial_pieces();
  test_transform_spatial_tensors_to_different_frame();
  test_modal_nodal_invertibility();
  test_apply_filter<filter_detail::gh_spacetime_vars_list>(0);
  test_apply_filter<filter_detail::gh_spacetime_vars_list>(5);
  test_apply_filter<filter_detail::sw_vars_list<Frame::Inertial>>(0);
  test_apply_filter<filter_detail::sw_vars_list<Frame::Inertial>>(5);
}
}  // namespace
}  // namespace ylm::TensorYlm
