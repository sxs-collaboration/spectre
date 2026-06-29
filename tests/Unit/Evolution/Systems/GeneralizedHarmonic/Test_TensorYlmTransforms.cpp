// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <random>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/SimpleSparseMatrix.hpp"
#include "DataStructures/Tensor/Structure.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/ApplyTensorYlmFilter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/TensorYlmTransforms.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/TensorYlm/ApplyFilter.tpp"
#include "NumericalAlgorithms/TensorYlm/CartToSphere.hpp"
#include "NumericalAlgorithms/TensorYlm/SphereToCart.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ylm::TensorYlm {
namespace {

struct TransformMatrices {
  SimpleSparseMatrix i{};
  SimpleSparseMatrix ii{};
  SimpleSparseMatrix ij{};
  SimpleSparseMatrix ijj{};
};

TransformMatrices make_cart_to_sphere_matrices(const size_t ell_max) {
  TransformMatrices matrices{};
  fill_cart_to_sphere<typename tnsr::i<DataVector, 3>::structure>(
      make_not_null(&matrices.i), ell_max,
      CoefficientNormalization::Spherepack);
  fill_cart_to_sphere<typename tnsr::ii<DataVector, 3>::structure>(
      make_not_null(&matrices.ii), ell_max,
      CoefficientNormalization::Spherepack);
  fill_cart_to_sphere<typename tnsr::ij<DataVector, 3>::structure>(
      make_not_null(&matrices.ij), ell_max,
      CoefficientNormalization::Spherepack);
  fill_cart_to_sphere<typename tnsr::ijj<DataVector, 3>::structure>(
      make_not_null(&matrices.ijj), ell_max,
      CoefficientNormalization::Spherepack);
  return matrices;
}

TransformMatrices make_sphere_to_cart_matrices(const size_t ell_max) {
  TransformMatrices matrices{};
  fill_sphere_to_cart<typename tnsr::i<DataVector, 3>::structure>(
      make_not_null(&matrices.i), ell_max,
      CoefficientNormalization::Spherepack);
  fill_sphere_to_cart<typename tnsr::ii<DataVector, 3>::structure>(
      make_not_null(&matrices.ii), ell_max,
      CoefficientNormalization::Spherepack);
  fill_sphere_to_cart<typename tnsr::ij<DataVector, 3>::structure>(
      make_not_null(&matrices.ij), ell_max,
      CoefficientNormalization::Spherepack);
  fill_sphere_to_cart<typename tnsr::ijj<DataVector, 3>::structure>(
      make_not_null(&matrices.ijj), ell_max,
      CoefficientNormalization::Spherepack);
  return matrices;
}

InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> identity_jacobian(
    const size_t size) {
  InverseJacobian<DataVector, 3, Frame::Inertial, Frame::Grid> jacobian{size,
                                                                        0.0};
  for (size_t i = 0; i < 3; ++i) {
    jacobian.get(i, i) = 1.0;
  }
  return jacobian;
}

// Scalars need no correction, but tensors must be transformed to the
// TensorYlm basis. This function transforms the input tensor coefficients
// to the TensorYlm basis.
template <typename TensorType>
void apply_tensor_ylm_basis_matrix_to_coefficients(
    const gsl::not_null<TensorType*> result, const TensorType& coefficients,
    const SimpleSparseMatrix& matrix, const size_t spectral_size,
    const size_t radial_extents) {
  if constexpr (TensorType::rank() == 0) {
    *result = coefficients;  // just copy the scalars
  } else {
    DataVector source{coefficients.size() * spectral_size};
    DataVector destination{result->size() * spectral_size};
    // Pack input tensor coefficients into a contiguous DataVector.
    // The transformation sparse matrices will act on the contiguous DataVector.
    for (size_t offset = 0; offset < radial_extents; ++offset) {
      for (size_t component = 0; component < coefficients.size(); ++component) {
        for (size_t coefficient_index = 0; coefficient_index < spectral_size;
             ++coefficient_index) {
          source[component * spectral_size + coefficient_index] =
              coefficients[component]
                          [coefficient_index * radial_extents + offset];
        }
      }

      // Zero destination before calling increment_multiply_on_right, which
      // does dest += matrix * source.
      destination = 0.0;
      const gsl::span<double> source_span{source.data(), source.size()};
      gsl::span<double> destination_span{destination.data(),
                                         destination.size()};
      matrix.increment_multiply_on_right(make_not_null(&destination_span), 0, 1,
                                         source_span, 0, 1);

      // Set result tensor components from the contiguous DataVector.
      for (size_t component = 0; component < result->size(); ++component) {
        for (size_t coefficient_index = 0; coefficient_index < spectral_size;
             ++coefficient_index) {
          (*result)[component][coefficient_index * radial_extents + offset] =
              destination[component * spectral_size + coefficient_index];
        }
      }
    }
  }
}

// Wrapper to transform a tag in a Variables to the TensorYlm basis.
template <typename Tag>
void apply_tensor_ylm_basis_matrix_to_tag(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        coefficients,
    const SimpleSparseMatrix& matrix, const size_t spectral_size,
    const size_t radial_extents) {
  auto transformed = get<Tag>(*coefficients);
  apply_tensor_ylm_basis_matrix_to_coefficients(make_not_null(&transformed),
                                                get<Tag>(*coefficients), matrix,
                                                spectral_size, radial_extents);
  get<Tag>(*coefficients) = transformed;
}

// Helper function to transform GH spatial variables to the TensorYlm basis.
void apply_tensor_ylm_basis_matrices(
    const gsl::not_null<
        Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>*>
        coefficients,
    const TransformMatrices& matrices, const size_t spectral_size,
    const size_t radial_extents) {
  tmpl::for_each<filter_detail::gh_spatial_vars_list<
      Frame::Grid>>([coefficients, &matrices, spectral_size,
                     radial_extents]<class Tag>(
                        const tmpl::type_<Tag> /*meta*/) {
    if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                 Symmetry<1>>) {
      apply_tensor_ylm_basis_matrix_to_tag<Tag>(coefficients, matrices.i,
                                                spectral_size, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<1, 1>>) {
      apply_tensor_ylm_basis_matrix_to_tag<Tag>(coefficients, matrices.ii,
                                                spectral_size, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1>>) {
      apply_tensor_ylm_basis_matrix_to_tag<Tag>(coefficients, matrices.ij,
                                                spectral_size, radial_extents);
    } else if constexpr (std::is_same_v<typename Tag::type::structure::symmetry,
                                        Symmetry<2, 1, 1>>) {
      apply_tensor_ylm_basis_matrix_to_tag<Tag>(coefficients, matrices.ijj,
                                                spectral_size, radial_extents);
    }
  });
}

// Wrapper for calling the function under test,
// gh_variables_to_tensor_ylm_coefficients(). This wrapper creates temporary
// storage, calls the function, then
// returns the transformed result.
Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> transform_gh_vars(
    const Variables<filter_detail::gh_spacetime_vars_list>& gh_vars,
    const Spherepack& spherepack, const size_t radial_extents,
    const TransformMatrices& matrices) {
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> result{
      spherepack.spectral_size() * radial_extents};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> temp_storage{
      spherepack.spectral_size() * radial_extents};
  gh_variables_to_tensor_ylm_coefficients(
      make_not_null(&result), make_not_null(&temp_storage), gh_vars,
      identity_jacobian(spherepack.physical_size() * radial_extents),
      matrices.i, matrices.ii, matrices.ij, matrices.ijj, spherepack,
      radial_extents);
  return result;
}

// Test 1: Transform random GH variables two different ways:
// a. Call gh_variables_to_tensor_ylm_coefficients(), the function under test.
// b. Do the transformation in a different way, step by step. The alternate
// path is less efficient (e.g. many more allocations).
// The test checks that both paths give the same result.
void test_against_alt_transform_path(
    const gsl::not_null<std::mt19937*> generator) {
  constexpr size_t ell_max = 4;
  constexpr size_t radial_extents = 3;
  const Spherepack spherepack{ell_max, ell_max};
  const auto matrices = make_cart_to_sphere_matrices(ell_max);
  const size_t physical_size = spherepack.physical_size() * radial_extents;
  const size_t spectral_size = spherepack.spectral_size() * radial_extents;
  const auto jacobian = identity_jacobian(physical_size);

  // Generate random GH variables
  Variables<filter_detail::gh_spacetime_vars_list> gh_vars{physical_size};
  std::uniform_real_distribution<> dist{-1.0, 1.0};
  for (size_t i = 0; i < gh_vars.size(); ++i) {
    gh_vars.data()[i] = dist(*generator);
  }

  // Do transformation method a.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> result{
      spectral_size};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> temp_storage{
      spectral_size};
  gh_variables_to_tensor_ylm_coefficients(
      make_not_null(&result), make_not_null(&temp_storage), gh_vars, jacobian,
      matrices.i, matrices.ii, matrices.ij, matrices.ijj, spherepack,
      radial_extents);

  // Do transformation method b.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      inertial_spatial_vars{physical_size};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> grid_spatial_vars{
      physical_size};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> expected{
      spectral_size};
  filter_detail::break_spacetime_vars_into_spatial_pieces(
      make_not_null(&inertial_spatial_vars), gh_vars);
  filter_detail::transform_spatial_tensors_to_different_frame_without_hessians(
      make_not_null(&grid_spatial_vars), inertial_spatial_vars, jacobian);
  filter_detail::nodal_to_modal_ylm(make_not_null(&expected), grid_spatial_vars,
                                    spherepack, radial_extents);
  apply_tensor_ylm_basis_matrices(make_not_null(&expected), matrices,
                                  spherepack.spectral_size(), radial_extents);

  CHECK_VARIABLES_APPROX(result, expected);
}

// Test 2: test that Minkowski spacetime has only l=0 modes.
void test_minkowski_has_only_constant_metric_modes() {
  constexpr size_t ell_max = 5;
  constexpr size_t radial_extents = 2;
  const Spherepack spherepack{ell_max, ell_max};
  const auto matrices = make_cart_to_sphere_matrices(ell_max);

  // Set up Minkowski GH variables
  Variables<filter_detail::gh_spacetime_vars_list> gh_vars{
      spherepack.physical_size() * radial_extents, 0.0};
  auto& metric =
      get<::gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(gh_vars);
  get<0, 0>(metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    metric.get(i + 1, i + 1) = 1.0;
  }

  // Check that all l>0 modes vanish in all tensor components.
  const auto result =
      transform_gh_vars(gh_vars, spherepack, radial_extents, matrices);
  const SpherepackIterator iterator{ell_max, ell_max, 1, false};
  tmpl::for_each<filter_detail::gh_spatial_vars_list<Frame::Grid>>(
      [&result, &iterator,
       radial_extents]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        const auto& tensor = get<Tag>(result);
        for (const auto& component : tensor) {
          for (SpherepackIterator it = iterator; it; ++it) {
            if (it.l() > 0) {
              for (size_t offset = 0; offset < radial_extents; ++offset) {
                CHECK(component[it() * radial_extents + offset] == approx(0.0));
              }
            }
          }
        }
      });

  // Pi and Phi are zero in Minkowski space, so also check that the spatial
  // pieces of Pi and Phi returned by transform_gh_vars vanish.
  tmpl::for_each<
      tmpl::list<filter_detail::Tags::Pi00<DataVector>,
                 filter_detail::Tags::Pik0<DataVector, 3, Frame::Grid>,
                 filter_detail::Tags::Pikj<DataVector, 3, Frame::Grid>,
                 filter_detail::Tags::Phik00<DataVector, 3, Frame::Grid>,
                 filter_detail::Tags::Phiki0<DataVector, 3, Frame::Grid>,
                 filter_detail::Tags::Phikij<DataVector, 3, Frame::Grid>>>(
      [&result]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        const auto& tensor = get<Tag>(result);
        for (const auto& component : tensor) {
          const DataVector expected_zero{component.size(), 0.0};
          CHECK_ITERABLE_CUSTOM_APPROX(component, expected_zero, approx);
        }
      });
}

// Test 3: check that a radial unit vector has vanishing m and mbar components
// and a radial component of 1.
void test_radial_vector_basis_component() {
  constexpr size_t ell_max = 5;
  constexpr size_t radial_extents = 1;
  const Spherepack spherepack{ell_max, ell_max};
  const auto matrices = make_cart_to_sphere_matrices(ell_max);
  const size_t physical_size = spherepack.physical_size() * radial_extents;

  // Set Pi and Phi, and SpacetimeMetric to zero, except set g_{ti} to
  // a radial unit vector.
  Variables<filter_detail::gh_spacetime_vars_list> gh_vars{physical_size, 0.0};
  auto& metric =
      get<::gr::Tags::SpacetimeMetric<DataVector, 3, Frame::Inertial>>(gh_vars);
  const auto theta_phi = spherepack.theta_phi_points();
  for (size_t s = 0; s < spherepack.physical_size(); ++s) {
    const double theta = theta_phi[0][s];
    const double phi = theta_phi[1][s];
    metric.get(1, 0)[s] = sin(theta) * cos(phi);
    metric.get(2, 0)[s] = sin(theta) * sin(phi);
    metric.get(3, 0)[s] = cos(theta);
  }

  const auto result =
      transform_gh_vars(gh_vars, spherepack, radial_extents, matrices);

  // Check that component 0 (l, or radial component), is 1, as it should be
  // since at each point, the unit vector has a radial component of 1 and
  // angular components vanishing.
  const auto& vector =
      get<filter_detail::Tags::Metrick0<DataVector, 3, Frame::Grid>>(result);
  const DataVector expected_radial = spherepack.phys_to_spec_all_offsets(
      DataVector{physical_size, 1.0}, radial_extents);
  CHECK_ITERABLE_CUSTOM_APPROX(vector.get(0), expected_radial, approx);

  // Check that components 1 and 2 (m and mbar, or angular components) vanish.
  const DataVector expected_zero{vector.get(1).size(), 0.0};
  CHECK_ITERABLE_CUSTOM_APPROX(vector.get(1), expected_zero, approx);
  CHECK_ITERABLE_CUSTOM_APPROX(vector.get(2), expected_zero, approx);
}

// Test 4: Start with modes, compute the original GH vars from them,
// then make sure that transform_gh_vars() gets back the original modes.
// Also explicitly verify that, when original modes are zero except for
// l=2, m=1, that the computed modes only have nonzero power in l=2; this is
// redundant in terms of checking for correctness but could help narrow down
// the issue if the round trip ever fails (wrong l=2, m=1 values vs.
// nonzero l!=2 modes).
void test_controlled_mode_roundtrip_and_power_selection() {
  constexpr size_t ell_max = 5;
  constexpr size_t radial_extents = 2;
  const Spherepack spherepack{ell_max, ell_max};
  const auto cart_to_sphere = make_cart_to_sphere_matrices(ell_max);
  const auto sphere_to_cart = make_sphere_to_cart_matrices(ell_max);
  const size_t spectral_size = spherepack.spectral_size() * radial_extents;
  const size_t physical_size = spherepack.physical_size() * radial_extents;

  // Choose one component of the metric to have some nonzero modes.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      tensor_ylm_coefficients{spectral_size, 0.0};
  auto& metric_modes =
      get<filter_detail::Tags::Metrickj<DataVector, 3, Frame::Grid>>(
          tensor_ylm_coefficients);
  SpherepackIterator iterator{ell_max, ell_max, 1, false};
  const size_t a21 =
      iterator.set(2, 1, SpherepackIterator::CoefficientArray::a)();
  const size_t b21 =
      iterator.set(2, 1, SpherepackIterator::CoefficientArray::b)();
  metric_modes.get(1, 2)[a21 * radial_extents] = 0.7;
  metric_modes.get(1, 2)[a21 * radial_extents + 1] = -0.4;
  metric_modes.get(1, 2)[b21 * radial_extents] = 0.2;
  metric_modes.get(1, 2)[b21 * radial_extents + 1] = 0.5;

  // Transform back from the TensorYlm basis to the Cartesian basis.
  auto cartesian_modal_coefficients = tensor_ylm_coefficients;
  apply_tensor_ylm_basis_matrices(make_not_null(&cartesian_modal_coefficients),
                                  sphere_to_cart, spherepack.spectral_size(),
                                  radial_extents);

  // Transform modal coefficients back to nodal coefficients.
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> grid_nodal_vars{
      physical_size};
  filter_detail::modal_to_nodal_ylm(make_not_null(&grid_nodal_vars),
                                    cartesian_modal_coefficients, spherepack,
                                    radial_extents);
  // Turn nodal spatial pieces in the Cartesian basis into spacetime GH
  // variables.
  Variables<filter_detail::gh_spacetime_vars_list> gh_vars{physical_size, 0.0};
  const Variables<filter_detail::gh_spatial_vars_list<Frame::Inertial>>
      inertial_nodal_vars{grid_nodal_vars.data(), grid_nodal_vars.size()};
  filter_detail::assemble_spacetime_vars_from_spatial_pieces(
      make_not_null(&gh_vars), inertial_nodal_vars);

  // Do the transform and check that the modes returned match the modes
  // this test started with.
  const auto result =
      transform_gh_vars(gh_vars, spherepack, radial_extents, cart_to_sphere);
  CHECK_VARIABLES_APPROX(result, tensor_ylm_coefficients);

  // Explicitly verify that power in the l == 2 mode is nonzero and power in
  // the other modes vanishes.
  DataVector power_by_l{ell_max + 1, 0.0};
  tmpl::for_each<filter_detail::gh_spatial_vars_list<Frame::Grid>>(
      [&result, &power_by_l,
       radial_extents]<class Tag>(const tmpl::type_<Tag> /*meta*/) {
        const auto& tensor = get<Tag>(result);
        for (const auto& component : tensor) {
          for (SpherepackIterator it{ell_max, ell_max, 1, false}; it; ++it) {
            for (size_t offset = 0; offset < radial_extents; ++offset) {
              power_by_l[it.l()] +=
                  square(component[it() * radial_extents + offset]);
            }
          }
        }
      });
  for (size_t l = 0; l <= ell_max; ++l) {
    if (l == 2) {
      CHECK(power_by_l[l] > 0.0);
    } else {
      CHECK(power_by_l[l] == approx(0.0));
    }
  }
}

// In debug builds, check all the asserts in TensorYlmTransforms.cpp.
#ifdef SPECTRE_DEBUG
void test_asserts() {
  constexpr size_t ell_max = 3;
  constexpr size_t radial_extents = 2;
  const Spherepack spherepack{ell_max, ell_max};
  const auto matrices = make_cart_to_sphere_matrices(ell_max);
  const Variables<filter_detail::gh_spacetime_vars_list> gh_vars{
      spherepack.physical_size() * radial_extents, 0.0};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> result{
      spherepack.spectral_size() * radial_extents};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> temp{
      spherepack.spectral_size() * radial_extents};
  CHECK_THROWS_WITH(
      gh_variables_to_tensor_ylm_coefficients(
          make_not_null(&result), make_not_null(&result), gh_vars,
          identity_jacobian(spherepack.physical_size() * radial_extents),
          matrices.i, matrices.ii, matrices.ij, matrices.ijj, spherepack,
          radial_extents),
      Catch::Matchers::ContainsSubstring("must not alias temp_storage"));

  const Variables<filter_detail::gh_spacetime_vars_list> wrong_size_gh_vars{
      spherepack.physical_size() * radial_extents + 1, 0.0};
  CHECK_THROWS_WITH(
      gh_variables_to_tensor_ylm_coefficients(
          make_not_null(&result), make_not_null(&temp), wrong_size_gh_vars,
          identity_jacobian(spherepack.physical_size() * radial_extents),
          matrices.i, matrices.ii, matrices.ij, matrices.ijj, spherepack,
          radial_extents),
      Catch::Matchers::ContainsSubstring("Expected GH variables"));

  const Spherepack spherepack_with_truncated_m{ell_max, ell_max - 1};
  const Variables<filter_detail::gh_spacetime_vars_list> gh_vars_truncated_m{
      spherepack_with_truncated_m.physical_size() * radial_extents, 0.0};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>>
      result_truncated_m{spherepack_with_truncated_m.spectral_size() *
                         radial_extents};
  Variables<filter_detail::gh_spatial_vars_list<Frame::Grid>> temp_truncated_m{
      spherepack_with_truncated_m.spectral_size() * radial_extents};
  CHECK_THROWS_WITH(
      gh_variables_to_tensor_ylm_coefficients(
          make_not_null(&result_truncated_m), make_not_null(&temp_truncated_m),
          gh_vars_truncated_m,
          identity_jacobian(spherepack_with_truncated_m.physical_size() *
                            radial_extents),
          matrices.i, matrices.ii, matrices.ij, matrices.ijj,
          spherepack_with_truncated_m, radial_extents),
      Catch::Matchers::ContainsSubstring("require m_max == l_max"));
}
#endif

// [[TimeOut, 20]]
SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.TensorYlmTransforms",
    "[Unit][NumericalAlgorithms]") {
  MAKE_GENERATOR(generator);
  test_against_alt_transform_path(make_not_null(&generator));
  test_minkowski_has_only_constant_metric_modes();
  test_radial_vector_basis_component();
  test_controlled_mode_roundtrip_and_power_selection();
#ifdef SPECTRE_DEBUG
  test_asserts();
#endif
}

}  // namespace
}  // namespace ylm::TensorYlm
