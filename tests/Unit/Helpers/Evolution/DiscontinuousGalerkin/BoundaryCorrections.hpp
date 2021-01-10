// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers::evolution::dg {
/// Indicate if the boundary correction should be zero when the solution is
/// smooth (should pretty much always be `Yes`)
enum class ZeroOnSmoothSolution { Yes, No };

namespace detail {
template <bool HasPrimitiveVars = false>
struct get_correction_primitive_vars_impl {
  template <typename BoundaryCorrection>
  using f = tmpl::list<>;
};

template <>
struct get_correction_primitive_vars_impl<true> {
  template <typename BoundaryCorrection>
  using f = typename BoundaryCorrection::dg_package_data_primitive_tags;
};

template <bool HasPrimitiveVars, typename BoundaryCorrection>
using get_correction_primitive_vars =
    typename get_correction_primitive_vars_impl<HasPrimitiveVars>::template f<
        BoundaryCorrection>;

template <bool HasPrimitiveVars = false>
struct get_system_primitive_vars_impl {
  template <typename System>
  using f = tmpl::list<>;
};

template <>
struct get_system_primitive_vars_impl<true> {
  template <typename System>
  using f = typename System::primitive_variables_tag::tags_list;
};

template <bool HasPrimitiveVars, typename System>
using get_system_primitive_vars = typename get_system_primitive_vars_impl<
    HasPrimitiveVars>::template f<System>;

template <typename BoundaryCorrection, typename... PackageTags,
          typename... FaceTags, typename... VolumeTags, size_t Dim>
void call_dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackageTags...>>*> package_data,
    const BoundaryCorrection& correction,
    const Variables<tmpl::list<FaceTags...>>& face_variables,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity) {
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity =
        dot_product(*mesh_velocity, unit_normal_covector);
  }
  correction.dg_package_data(
      make_not_null(&get<PackageTags>(*package_data))...,
      get<FaceTags>(face_variables)..., unit_normal_covector, mesh_velocity,
      normal_dot_mesh_velocity, get<VolumeTags>(volume_data)...);
}

template <typename BoundaryCorrection, typename... BoundaryCorrectionTags,
          typename... PackageTags>
void call_dg_boundary_terms(
    const gsl::not_null<Variables<tmpl::list<BoundaryCorrectionTags...>>*>
        boundary_corrections,
    const BoundaryCorrection& correction,
    const Variables<tmpl::list<PackageTags...>>& interior_package_data,
    const Variables<tmpl::list<PackageTags...>>& exterior_package_data,
    const ::dg::Formulation dg_formulation) {
  correction.dg_boundary_terms(
      make_not_null(&get<BoundaryCorrectionTags>(*boundary_corrections))...,
      get<PackageTags>(interior_package_data)...,
      get<PackageTags>(exterior_package_data)..., dg_formulation);
}

template <typename System, typename BoundaryCorrection, size_t FaceDim,
          typename... VolumeTags>
void test_boundary_correction_impl(
    const BoundaryCorrection& correction, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const bool use_moving_mesh, const ::dg::Formulation dg_formulation,
    const ZeroOnSmoothSolution zero_on_smooth_solution) {
  CAPTURE(use_moving_mesh);
  CAPTURE(dg_formulation);
  CAPTURE(FaceDim);
  using variables_tags = typename System::variables_tag::tags_list;
  using primitive_variables_tags = detail::get_system_primitive_vars<
      System::has_primitive_and_conservative_vars, System>;
  using flux_variables = typename System::flux_variables;
  using flux_tags =
      db::wrap_tags_in<::Tags::Flux, flux_variables, tmpl::size_t<FaceDim + 1>,
                       Frame::Inertial>;
  using temporary_tags =
      typename System::compute_volume_time_derivative_terms::temporary_tags;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;

  using dg_package_field_tags =
      typename BoundaryCorrection::dg_package_field_tags;
  using package_temporary_tags =
      typename BoundaryCorrection::dg_package_data_temporary_tags;
  using package_primitive_tags = detail::get_correction_primitive_vars<
      System::has_primitive_and_conservative_vars, BoundaryCorrection>;

  // Check that the temporary tags needed on the boundary
  // (package_temporary_tags) are listed as temporary tags for the volume time
  // derivative computation (temporary_tags).
  static_assert(
      std::is_same_v<
          tmpl::list_difference<package_temporary_tags, temporary_tags>,
          tmpl::list<>>,
      "There are temporary tags needed by the boundary correction that are not "
      "computed as temporary tags by the system");

  // Check that the primitive tags needed on the boundary
  // (package_primitive_tags) are listed as the primitive tags in
  // the system (primitive_variables_tags).
  static_assert(
      std::is_same_v<tmpl::list_difference<package_primitive_tags,
                                           primitive_variables_tags>,
                     tmpl::list<>>,
      "There are primitive tags needed by the boundary correction that are not "
      "listed in the system as being primitive variables");

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  std::uniform_real_distribution<> pos_neg_dist(-1.0, 1.0);
  DataVector used_for_size{face_mesh.number_of_grid_points()};

  std::optional<tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>
      mesh_velocity{};
  if (use_moving_mesh) {
    mesh_velocity = make_with_random_values<
        tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  }

  Variables<dg_package_field_tags> interior_package_data{used_for_size.size()};
  const auto interior_fields_on_face = make_with_random_values<
      Variables<tmpl::append<variables_tags, flux_tags, package_temporary_tags,
                             package_primitive_tags>>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);

  Variables<dg_package_field_tags> exterior_package_data{used_for_size.size()};
  const auto exterior_fields_on_face = make_with_random_values<
      Variables<tmpl::append<variables_tags, flux_tags, package_temporary_tags,
                             package_primitive_tags>>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);

  // Compute the interior and exterior normal vectors so they are pointing in
  // opposite directions.
  auto interior_unit_normal_covector = make_with_random_values<
      tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>>(
      make_not_null(&gen), make_not_null(&pos_neg_dist), used_for_size);
  const Scalar<DataVector> interior_normal_magnitude =
      magnitude(interior_unit_normal_covector);
  for (auto& t : interior_unit_normal_covector) {
    t /= get(interior_normal_magnitude);
  }
  auto exterior_unit_normal_covector = interior_unit_normal_covector;
  for (auto& t : exterior_unit_normal_covector) {
    t *= -1.0;
  }

  call_dg_package_data(make_not_null(&interior_package_data), correction,
                       interior_fields_on_face, volume_data,
                       interior_unit_normal_covector, mesh_velocity);
  call_dg_package_data(make_not_null(&exterior_package_data), correction,
                       exterior_fields_on_face, volume_data,
                       exterior_unit_normal_covector, mesh_velocity);

  Variables<dt_variables_tags> boundary_corrections{
      face_mesh.number_of_grid_points()};
  call_dg_boundary_terms(make_not_null(&boundary_corrections), correction,
                         interior_package_data, exterior_package_data,
                         dg_formulation);

  if (dg_formulation == ::dg::Formulation::StrongInertial) {
    // The strong form should be (WeakForm - (n_i F^i)_{interior}).
    // Since we also test conservation for the weak form we just need to test
    // that the strong form satisfies the above definition.

    Variables<dt_variables_tags> expected_boundary_corrections{
        face_mesh.number_of_grid_points()};
    call_dg_boundary_terms(make_not_null(&expected_boundary_corrections),
                           correction, interior_package_data,
                           exterior_package_data,
                           ::dg::Formulation::WeakInertial);

    tmpl::for_each<flux_variables>([&interior_package_data,
                                    &expected_boundary_corrections](
                                       auto flux_variable_tag_v) noexcept {
      using flux_variable_tag = tmpl::type_from<decltype(flux_variable_tag_v)>;
      using normal_dot_flux_tag = ::Tags::NormalDotFlux<flux_variable_tag>;
      using dt_tag = ::Tags::dt<flux_variable_tag>;
      const auto& normal_dot_flux =
          get<normal_dot_flux_tag>(interior_package_data);
      auto& expected_boundary_correction =
          get<dt_tag>(expected_boundary_corrections);
      for (size_t tensor_index = 0;
           tensor_index < expected_boundary_correction.size(); ++tensor_index) {
        expected_boundary_correction[tensor_index] -=
            normal_dot_flux[tensor_index];
      }
    });
    {
      INFO("Check weak and strong boundary terms match.");
      CHECK(boundary_corrections == expected_boundary_corrections);
    }

    if (zero_on_smooth_solution == ZeroOnSmoothSolution::Yes) {
      INFO(
          "Testing that if the solution is the same on both sides the "
          "StrongInertial correction is identically zero.");
      Variables<dg_package_field_tags> interior_package_data_opposite_signs{
          used_for_size.size()};
      call_dg_package_data(make_not_null(&interior_package_data_opposite_signs),
                           correction, interior_fields_on_face, volume_data,
                           exterior_unit_normal_covector, mesh_velocity);
      Variables<dt_variables_tags> zero_boundary_correction{
          face_mesh.number_of_grid_points()};
      call_dg_boundary_terms(make_not_null(&zero_boundary_correction),
                             correction, interior_package_data,
                             interior_package_data_opposite_signs,
                             ::dg::Formulation::StrongInertial);
      Variables<dt_variables_tags> expected_zero_boundary_correction{
          face_mesh.number_of_grid_points(), 0.0};
      tmpl::for_each<dt_variables_tags>([&expected_zero_boundary_correction,
                                         &zero_boundary_correction](
                                            auto dt_variables_tag_v) noexcept {
        using dt_variables_tag = tmpl::type_from<decltype(dt_variables_tag_v)>;
        const std::string tag_name = db::tag_name<dt_variables_tag>();
        CAPTURE(tag_name);
        Approx custom_approx = Approx::custom().epsilon(1.e-12).scale(1.0);
        CHECK_ITERABLE_CUSTOM_APPROX(
            get<dt_variables_tag>(zero_boundary_correction),
            get<dt_variables_tag>(expected_zero_boundary_correction),
            custom_approx);
      });
    }
  } else if (dg_formulation == ::dg::Formulation::WeakInertial) {
    INFO(
        "Checking that swapping the two sides results in an overall minus "
        "sign.");
    Variables<dt_variables_tags> reverse_side_boundary_corrections{
        face_mesh.number_of_grid_points()};
    call_dg_boundary_terms(make_not_null(&reverse_side_boundary_corrections),
                           correction, exterior_package_data,
                           interior_package_data, dg_formulation);
    // Check that the flux leaving one element equals the flux entering its
    // neighbor, i.e., F*(interior, exterior) == -F*(exterior, interior)
    reverse_side_boundary_corrections *= -1.0;
    tmpl::for_each<flux_variables>([&boundary_corrections,
                                    &reverse_side_boundary_corrections](
                                       auto flux_variable_tag_v) {
      using flux_variable_tag = tmpl::type_from<decltype(flux_variable_tag_v)>;
      const std::string tag_name = db::tag_name<flux_variable_tag>();
      CAPTURE(tag_name);
      CHECK_ITERABLE_APPROX(
          get<::Tags::dt<flux_variable_tag>>(reverse_side_boundary_corrections),
          get<::Tags::dt<flux_variable_tag>>(boundary_corrections));
    });
  } else {
    ERROR("DG formulation must be StrongInertial or WeakInertial, not "
          << dg_formulation);
  }
}
}  // namespace detail

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Checks that the boundary correction is conservative and that for
 * smooth solutions the strong-form correction is zero.
 */
template <typename System, typename BoundaryCorrection, size_t FaceDim,
          typename... VolumeTags>
void test_boundary_correction_conservation(
    const BoundaryCorrection& correction, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const ZeroOnSmoothSolution zero_on_smooth_solution =
        ZeroOnSmoothSolution::Yes) {
  for (const auto use_moving_mesh : {true, false}) {
    for (const auto& dg_formulation :
         {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
      detail::test_boundary_correction_impl<System>(
          correction, face_mesh, volume_data, use_moving_mesh, dg_formulation,
          zero_on_smooth_solution);
    }
  }
}

namespace detail {
template <typename ConversionClassList, typename VariablesTags,
          typename BoundaryCorrection, size_t FaceDim, typename... FaceTags,
          typename... VolumeTags, typename... DgPackageDataTags>
void test_with_python(
    const std::string& python_module,
    const std::array<
        std::string,
        tmpl::size<typename BoundaryCorrection::dg_package_field_tags>::value>&
        python_dg_package_data_functions,
    const std::array<std::string, tmpl::size<VariablesTags>::value>&
        python_dg_boundary_terms_functions,
    const BoundaryCorrection& correction, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const bool use_moving_mesh, const ::dg::Formulation dg_formulation,
    const double epsilon, tmpl::list<FaceTags...> /*meta*/,
    tmpl::list<DgPackageDataTags...> /*meta*/) {
  CAPTURE(face_mesh);
  CAPTURE(dg_formulation);
  CAPTURE(use_moving_mesh);
  REQUIRE(face_mesh.number_of_grid_points() >= 1);
  using dg_package_field_tags =
      typename BoundaryCorrection::dg_package_field_tags;

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.0, 1.0);
  DataVector used_for_size{face_mesh.number_of_grid_points()};

  Variables<dg_package_field_tags> package_data{used_for_size.size()};
  const auto fields_on_face =
      make_with_random_values<Variables<tmpl::list<FaceTags...>>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  auto unit_normal_covector = make_with_random_values<
      tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto normal_magnitude = magnitude(unit_normal_covector);
  for (auto& component : unit_normal_covector) {
    component /= get(normal_magnitude);
  }
  std::optional<tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>
      mesh_velocity{};
  if (use_moving_mesh) {
    mesh_velocity = make_with_random_values<
        tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  }
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity =
        dot_product(*mesh_velocity, unit_normal_covector);
  }

  // Call C++ implementation of dg_package_data
  call_dg_package_data(make_not_null(&package_data), correction, fields_on_face,
                       volume_data, unit_normal_covector, mesh_velocity);

  // Call python implementation of dg_package_data
  size_t function_name_index = 0;
  tmpl::for_each<dg_package_field_tags>(
      [epsilon, &fields_on_face, &function_name_index, &mesh_velocity,
       &normal_dot_mesh_velocity, &package_data,
       &python_dg_package_data_functions, &python_module, &unit_normal_covector,
       &volume_data](auto package_data_tag_v) {
        // avoid compiler warnings if there isn't any volume data
        (void)volume_data;
        INFO("Testing package data");
        using package_data_tag = tmpl::type_from<decltype(package_data_tag_v)>;
        using ResultType = typename package_data_tag::type;
        try {
          CAPTURE(python_module);
          CAPTURE(
              gsl::at(python_dg_package_data_functions, function_name_index));
          const auto python_result =
              pypp::call<ResultType, ConversionClassList>(
                  python_module,
                  gsl::at(python_dg_package_data_functions,
                          function_name_index),
                  get<FaceTags>(fields_on_face)..., unit_normal_covector,
                  mesh_velocity, normal_dot_mesh_velocity,
                  get<VolumeTags>(volume_data)...);
          CHECK_ITERABLE_CUSTOM_APPROX(
              get<package_data_tag>(package_data), python_result,
              Approx::custom().epsilon(epsilon).scale(1.0));
        } catch (const std::exception& e) {
          INFO("On line " << __LINE__ << " Python call to "
                          << gsl::at(python_dg_package_data_functions,
                                     function_name_index)
                          << " in module " << python_module
                          << " failed: " << e.what());
          REQUIRE(false);
        }
        ++function_name_index;
      });

  // Now we need to check the dg_boundary_terms function.
  const auto interior_package_data =
      make_with_random_values<Variables<dg_package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  const auto exterior_package_data =
      make_with_random_values<Variables<dg_package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  // We don't need to prefix the VariablesTags with anything because we are not
  // interacting with any code that cares about what the tags are, just that the
  // types matched the evolved variables.
  Variables<VariablesTags> boundary_corrections{
      face_mesh.number_of_grid_points()};

  // Call C++ implementation of dg_boundary_terms
  call_dg_boundary_terms(make_not_null(&boundary_corrections), correction,
                         interior_package_data, exterior_package_data,
                         dg_formulation);

  // Call python implementation of dg_boundary_terms
  function_name_index = 0;
  tmpl::for_each<VariablesTags>([&boundary_corrections, &dg_formulation,
                                 epsilon, &exterior_package_data,
                                 &function_name_index, &interior_package_data,
                                 &python_dg_boundary_terms_functions,
                                 &python_module](auto package_data_tag_v) {
    INFO("Testing boundary terms");
    using boundary_correction_tag =
        tmpl::type_from<decltype(package_data_tag_v)>;
    using ResultType = typename boundary_correction_tag::type;
    try {
      CAPTURE(python_module);
      const std::string& python_function =
          gsl::at(python_dg_boundary_terms_functions, function_name_index);
      CAPTURE(python_function);
      // Make explicitly depend on tag type to avoid type deduction issues with
      // GCC7
      const typename boundary_correction_tag::type python_result =
          pypp::call<ResultType>(
              python_module, python_function,
              get<DgPackageDataTags>(interior_package_data)...,
              get<DgPackageDataTags>(exterior_package_data)...,
              dg_formulation == ::dg::Formulation::StrongInertial);
      CHECK_ITERABLE_CUSTOM_APPROX(
          get<boundary_correction_tag>(boundary_corrections), python_result,
          Approx::custom().epsilon(epsilon).scale(1.0));
    } catch (const std::exception& e) {
      INFO("On line " << __LINE__ << " Python call to "
                      << gsl::at(python_dg_boundary_terms_functions,
                                 function_name_index)
                      << " in module " << python_module
                      << " failed: " << e.what());
      REQUIRE(false);
    }
    ++function_name_index;
  });
}
}  // namespace detail

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Tests that the `dg_package_data` and `dg_boundary_terms` functions
 * agree with the python implementation.
 *
 * The variables are filled with random numbers between zero and one before
 * being passed to the implementations. If in the future we need support for
 * negative numbers we can add the ability to specify a single range for all
 * random numbers or each individually.
 *
 * Please note the following:
 * - The `pypp::SetupLocalPythonEnvironment` must be created before the
 *   `test_boundary_correction_with_python` can be called.
 * - The `dg_formulation` is passed as a bool `use_strong_form` to the python
 *   functions since we don't want to rely on python bindings for the enum.
 * - You can convert custom types using the `ConversionClassList` template
 *   parameter, which is then passed to `pypp::call()`. This allows you to,
 *   e.g., convert an equation of state into an array locally in a test file.
 * - There must be one python function to compute the packaged data for each tag
 *   in `dg_package_field_tags`
 * - There must be one python function to compute the boundary correction for
 *   each tag in `System::variables_tag`
 * - The arguments to the python functions for computing the packaged data are
 *   the same as the arguments for the C++ `dg_package_data` function, excluding
 *   the `gsl::not_null` arguments.
 * - The arguments to the python functions for computing the boundary
 *   corrections are the same as the arguments for the C++ `dg_boundary_terms`
 *   function, excluding the `gsl::not_null` arguments.
 */
template <typename System, typename ConversionClassList = tmpl::list<>,
          typename BoundaryCorrection, size_t FaceDim, typename... VolumeTags>
void test_boundary_correction_with_python(
    const std::string& python_module,
    const std::array<
        std::string,
        tmpl::size<typename BoundaryCorrection::dg_package_field_tags>::value>&
        python_dg_package_data_functions,
    const std::array<
        std::string,
        tmpl::size<typename System::variables_tag::tags_list>::value>&
        python_dg_boundary_terms_functions,
    const BoundaryCorrection& correction, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const double epsilon = 1.0e-12) {
  static_assert(std::is_final_v<std::decay_t<BoundaryCorrection>>,
                "All boundary correction classes must be marked `final`.");
  using package_temporary_tags =
      typename BoundaryCorrection::dg_package_data_temporary_tags;
  using package_primitive_tags = detail::get_correction_primitive_vars<
      System::has_primitive_and_conservative_vars, BoundaryCorrection>;
  using variables_tags = typename System::variables_tag::tags_list;
  using flux_variables = typename System::flux_variables;
  using flux_tags =
      db::wrap_tags_in<::Tags::Flux, flux_variables, tmpl::size_t<FaceDim + 1>,
                       Frame::Inertial>;

  for (const auto use_moving_mesh : {false, true}) {
    for (const auto dg_formulation :
         {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
      detail::test_with_python<ConversionClassList, variables_tags>(
          python_module, python_dg_package_data_functions,
          python_dg_boundary_terms_functions, correction, face_mesh,
          volume_data, use_moving_mesh, dg_formulation, epsilon,
          tmpl::append<variables_tags, flux_tags, package_temporary_tags,
                       package_primitive_tags>{},
          typename BoundaryCorrection::dg_package_field_tags{});
    }
  }
}
}  // namespace TestHelpers::evolution::dg
