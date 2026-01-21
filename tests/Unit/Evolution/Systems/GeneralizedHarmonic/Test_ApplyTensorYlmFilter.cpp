// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/Test_ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/ApplyTensorYlmFilter.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/TensorYlmFilter.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace gh {
namespace {
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<Filters::Filter, tmpl::list<TensorYlmFilter>>>;
  };
};

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
  ylm::TensorYlm::filter_detail::modal_to_nodal_ylm(
      make_not_null(&nodal_vars), modal_vars, ylm, radial_extents);
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> test_modal_vars(
      spectral_mesh_size, 0.0);
  ylm::TensorYlm::filter_detail::nodal_to_modal_ylm(
      make_not_null(&test_modal_vars), nodal_vars, ylm, radial_extents);

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

// Debug builds are timing out slightly, so increase the timeout.
// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.ApplyTensorYlmFilter",
    "[NumericalAlgorithms][Unit]") {
  register_factory_classes_with_charm<Metavariables>();

  const auto created_filter = TestHelpers::test_creation<
      std::unique_ptr<Filters::Filter>, Metavariables>(
      "TensorYlmFilter:\n"
      "  NumModesToKill: 2\n"
      "  HalfPower: 5");
  const auto& concrete_filter =
      dynamic_cast<const TensorYlmFilter&>(*created_filter);
  CHECK(concrete_filter == TensorYlmFilter{2, 5});
  CHECK(concrete_filter.blocks_to_filter() == std::nullopt);

  const auto deserialized_filter = serialize_and_deserialize(created_filter);
  CHECK(dynamic_cast<const TensorYlmFilter&>(*deserialized_filter) ==
        concrete_filter);

  test_break_spacetime_vars_into_spatial_pieces();
  test_transform_spatial_tensors_to_different_frame();
  test_modal_nodal_invertibility();

  const auto apply_filter = [](const auto vars_nodal, const auto vars_storage,
                               const auto& jac_inertial_to_grid,
                               const auto& jac_grid_to_inertial,
                               const auto& filter_matrices,
                               const size_t ell_max,
                               const size_t radial_extents) {
    apply_tensor_ylm_filter(
        vars_nodal, vars_storage, jac_inertial_to_grid, jac_grid_to_inertial,
        filter_matrices.scalar, filter_matrices.i, filter_matrices.ii,
        filter_matrices.ij, filter_matrices.kii, ell_max, radial_extents);
  };
  ylm::TensorYlm::test_apply_filter<filter_detail::gh_spacetime_vars_list,
                                    true>(0, apply_filter);
  ylm::TensorYlm::test_apply_filter<filter_detail::gh_spacetime_vars_list,
                                    true>(5, apply_filter);
}
}  // namespace
}  // namespace gh
