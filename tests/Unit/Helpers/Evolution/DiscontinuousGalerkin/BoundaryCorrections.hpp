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
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/NormalVectors.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits/CreateHasTypeAlias.hpp"

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
          typename... FaceTags, typename... VolumeTags,
          typename... FaceTagsToForward, size_t Dim>
double call_dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackageTags...>>*> package_data,
    const BoundaryCorrection& correction,
    const Variables<tmpl::list<FaceTags...>>& face_variables,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    tmpl::list<FaceTagsToForward...> /*meta*/) {
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity =
        dot_product(*mesh_velocity, unit_normal_covector);
  }
  const double max_speed = correction.dg_package_data(
      make_not_null(&get<PackageTags>(*package_data))...,
      get<FaceTagsToForward>(face_variables)..., unit_normal_covector,
      mesh_velocity, normal_dot_mesh_velocity, get<VolumeTags>(volume_data)...);
  return max_speed;
}

template <typename BoundaryCorrection, typename... PackageTags,
          typename... FaceTags, typename... VolumeTags,
          typename... FaceTagsToForward, size_t Dim>
double call_dg_package_data(
    const gsl::not_null<Variables<tmpl::list<PackageTags...>>*> package_data,
    const BoundaryCorrection& correction,
    const Variables<tmpl::list<FaceTags...>>& face_variables,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_covector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& unit_normal_vector,
    const std::optional<tnsr::I<DataVector, Dim, Frame::Inertial>>&
        mesh_velocity,
    tmpl::list<FaceTagsToForward...> /*meta*/) {
  std::optional<Scalar<DataVector>> normal_dot_mesh_velocity{};
  if (mesh_velocity.has_value()) {
    normal_dot_mesh_velocity =
        dot_product(*mesh_velocity, unit_normal_covector);
  }
  const double max_speed = correction.dg_package_data(
      make_not_null(&get<PackageTags>(*package_data))...,
      get<FaceTagsToForward>(face_variables)..., unit_normal_covector,
      unit_normal_vector, mesh_velocity, normal_dot_mesh_velocity,
      get<VolumeTags>(volume_data)...);
  return max_speed;
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
          typename... VolumeTags, typename... RangeTags>
void test_boundary_correction_conservation_impl(
    const BoundaryCorrection& correction_in, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
    const bool use_moving_mesh, const ::dg::Formulation dg_formulation,
    const ZeroOnSmoothSolution zero_on_smooth_solution) {
  CAPTURE(use_moving_mesh);
  CAPTURE(dg_formulation);
  CAPTURE(FaceDim);
  constexpr bool curved_background =
      detail::has_inverse_spatial_metric_tag_v<System>;
  CAPTURE(curved_background);

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

  const auto correction_base_ptr =
      serialize_and_deserialize(correction_in.get_clone());
  const auto& correction =
      dynamic_cast<const BoundaryCorrection&>(*correction_base_ptr);

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
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  DataVector used_for_size{face_mesh.number_of_grid_points()};

  std::optional<tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>
      mesh_velocity{};
  if (use_moving_mesh) {
    mesh_velocity = make_with_random_values<
        tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  }

  using face_tags =
      tmpl::append<variables_tags, flux_tags, package_temporary_tags,
                   package_primitive_tags>;
  using face_tags_with_curved_background = tmpl::conditional_t<
      curved_background,
      tmpl::remove_duplicates<tmpl::push_back<
          face_tags, typename detail::inverse_spatial_metric_tag<
                         curved_background>::template f<System>>>,
      face_tags>;

  // Fill all fields with random values in [-1,1), then, for each tag with a
  // specified range, overwrite with new random values in [min,max)
  Variables<dg_package_field_tags> interior_package_data{used_for_size.size()};
  auto interior_fields_on_face =
      make_with_random_values<Variables<face_tags_with_curved_background>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  tmpl::for_each<tmpl::list<RangeTags...>>([&gen, &interior_fields_on_face,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::array<double, 2>& range = tuples::get<Tags::Range<tag>>(ranges);
    std::uniform_real_distribution<> local_dist(range[0], range[1]);
    fill_with_random_values(make_not_null(&get<tag>(interior_fields_on_face)),
                            make_not_null(&gen), make_not_null(&local_dist));
  });

  // Same as above but now for external data
  Variables<dg_package_field_tags> exterior_package_data{used_for_size.size()};
  auto exterior_fields_on_face =
      make_with_random_values<Variables<face_tags_with_curved_background>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  tmpl::for_each<tmpl::list<RangeTags...>>([&gen, &exterior_fields_on_face,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::array<double, 2>& range = tuples::get<Tags::Range<tag>>(ranges);
    std::uniform_real_distribution<> local_dist(range[0], range[1]);
    fill_with_random_values(make_not_null(&get<tag>(exterior_fields_on_face)),
                            make_not_null(&gen), make_not_null(&local_dist));
  });

  // Compute the interior and exterior normal vectors so they are pointing in
  // opposite directions.
  auto interior_unit_normal_covector = make_with_random_values<
      tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>
      interior_unit_normal_vector{};

  tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>
      exterior_unit_normal_covector;
  tnsr::I<DataVector, FaceDim + 1, Frame::Inertial>
      exterior_unit_normal_vector{};
  if constexpr (not curved_background) {
    const Scalar<DataVector> interior_normal_magnitude =
        magnitude(interior_unit_normal_covector);
    for (auto& t : interior_unit_normal_covector) {
      t /= get(interior_normal_magnitude);
    }
    exterior_unit_normal_covector = interior_unit_normal_covector;
    for (auto& t : exterior_unit_normal_covector) {
      t *= -1.0;
    }
  } else {
    using inv_spatial_metric = typename detail::inverse_spatial_metric_tag<
        curved_background>::template f<System>;
    exterior_unit_normal_covector = interior_unit_normal_covector;
    for (auto& t : exterior_unit_normal_covector) {
      t *= -1.0;
    }
    detail::adjust_inverse_spatial_metric(
        make_not_null(&get<inv_spatial_metric>(interior_fields_on_face)));

    // make the exterior inverse spatial metric be close to the interior one
    // since the solution should be smooth. We can't change too much because we
    // want the spatial metric to have diagonal terms much larger than the
    // off-diagonal terms.
    std::uniform_real_distribution<> inv_metric_change_dist(0.999, 1.0);
    for (size_t i = 0; i < FaceDim + 1; ++i) {
      for (size_t j = i; j < FaceDim + 1; ++j) {
        get<inv_spatial_metric>(exterior_fields_on_face).get(i, j) =
            inv_metric_change_dist(gen) *
            get<inv_spatial_metric>(interior_fields_on_face).get(i, j);
      }
    }
    detail::normalize_vector_and_covector(
        make_not_null(&interior_unit_normal_covector),
        make_not_null(&interior_unit_normal_vector),
        get<inv_spatial_metric>(interior_fields_on_face));
    detail::normalize_vector_and_covector(
        make_not_null(&exterior_unit_normal_covector),
        make_not_null(&exterior_unit_normal_vector),
        get<inv_spatial_metric>(exterior_fields_on_face));
  }

  if constexpr (curved_background) {
    call_dg_package_data(
        make_not_null(&interior_package_data), correction,
        interior_fields_on_face, volume_data, interior_unit_normal_covector,
        interior_unit_normal_vector, mesh_velocity, face_tags{});
    call_dg_package_data(
        make_not_null(&exterior_package_data), correction,
        exterior_fields_on_face, volume_data, exterior_unit_normal_covector,
        exterior_unit_normal_vector, mesh_velocity, face_tags{});
  } else {
    call_dg_package_data(make_not_null(&interior_package_data), correction,
                         interior_fields_on_face, volume_data,
                         interior_unit_normal_covector, mesh_velocity,
                         face_tags{});
    call_dg_package_data(make_not_null(&exterior_package_data), correction,
                         exterior_fields_on_face, volume_data,
                         exterior_unit_normal_covector, mesh_velocity,
                         face_tags{});
  }

  Variables<dt_variables_tags> boundary_corrections{
      face_mesh.number_of_grid_points()};
  call_dg_boundary_terms(make_not_null(&boundary_corrections), correction,
                         interior_package_data, exterior_package_data,
                         dg_formulation);

  if (dg_formulation == ::dg::Formulation::StrongInertial) {
    // The strong form should be (WeakForm - (n_i F^i)_{interior}).
    // Since we also test conservation for the weak form we just need to test
    // that the strong form satisfies the above definition.

    if constexpr (curved_background) {
      call_dg_package_data(
          make_not_null(&interior_package_data), correction,
          interior_fields_on_face, volume_data, interior_unit_normal_covector,
          interior_unit_normal_vector, mesh_velocity, face_tags{});
      call_dg_package_data(
          make_not_null(&exterior_package_data), correction,
          exterior_fields_on_face, volume_data, exterior_unit_normal_covector,
          exterior_unit_normal_vector, mesh_velocity, face_tags{});
    } else {
      call_dg_package_data(make_not_null(&interior_package_data), correction,
                           interior_fields_on_face, volume_data,
                           interior_unit_normal_covector, mesh_velocity,
                           face_tags{});
      call_dg_package_data(make_not_null(&exterior_package_data), correction,
                           exterior_fields_on_face, volume_data,
                           exterior_unit_normal_covector, mesh_velocity,
                           face_tags{});
    }

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
      if constexpr (curved_background) {
        // If the solution is the same on both sides then the interior and
        // exterior normal vectors only differ by a sign.
        for (size_t i = 0; i < FaceDim + 1; ++i) {
          exterior_unit_normal_vector.get(i) =
              -interior_unit_normal_vector.get(i);
          exterior_unit_normal_covector.get(i) =
              -interior_unit_normal_covector.get(i);
        }
        call_dg_package_data(
            make_not_null(&interior_package_data_opposite_signs), correction,
            interior_fields_on_face, volume_data, exterior_unit_normal_covector,
            exterior_unit_normal_vector, mesh_velocity, face_tags{});
      } else {
        call_dg_package_data(
            make_not_null(&interior_package_data_opposite_signs), correction,
            interior_fields_on_face, volume_data, exterior_unit_normal_covector,
            mesh_velocity, face_tags{});
      }
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
          typename... VolumeTags, typename... RangeTags>
void test_boundary_correction_conservation(
    const BoundaryCorrection& correction, const Mesh<FaceDim>& face_mesh,
    const tuples::TaggedTuple<VolumeTags...>& volume_data,
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
    const ZeroOnSmoothSolution zero_on_smooth_solution =
        ZeroOnSmoothSolution::Yes) {
  for (const auto use_moving_mesh : {true, false}) {
    for (const auto& dg_formulation :
         {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
      detail::test_boundary_correction_conservation_impl<System>(
          correction, face_mesh, volume_data, ranges, use_moving_mesh,
          dg_formulation, zero_on_smooth_solution);
    }
  }
}

namespace detail {
template <typename System, typename ConversionClassList, typename VariablesTags,
          typename BoundaryCorrection, size_t FaceDim, typename... FaceTags,
          typename... VolumeTags, typename... RangeTags,
          typename... DgPackageDataTags>
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
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
    const bool use_moving_mesh, const ::dg::Formulation dg_formulation,
    const double epsilon, tmpl::list<FaceTags...> /*meta*/,
    tmpl::list<DgPackageDataTags...> /*meta*/) {
  CAPTURE(face_mesh);
  CAPTURE(dg_formulation);
  CAPTURE(use_moving_mesh);
  REQUIRE(face_mesh.number_of_grid_points() >= 1);
  constexpr bool curved_background =
      detail::has_inverse_spatial_metric_tag_v<System>;
  using dg_package_field_tags =
      typename BoundaryCorrection::dg_package_field_tags;

  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-1.0, 1.0);
  DataVector used_for_size{face_mesh.number_of_grid_points()};

  using face_tags = tmpl::list<FaceTags...>;
  using face_tags_with_curved_background = tmpl::conditional_t<
      curved_background,
      tmpl::remove_duplicates<tmpl::push_back<
          face_tags, typename TestHelpers::evolution::dg::detail::
                         inverse_spatial_metric_tag<
                             curved_background>::template f<System>>>,
      face_tags>;

  // Sanity check: we apply the same ranges to the random inputs of the
  // `dg_package_data` function and the `dg_boundary_terms` function. So
  // first we check that each range tag appears in at least one of these
  // function's arguments.
  using range_tags_list = tmpl::list<RangeTags...>;
  using ranges_for_dg_boundary_terms_only =
      tmpl::list_difference<range_tags_list, face_tags_with_curved_background>;
  using ranges_unused = tmpl::list_difference<ranges_for_dg_boundary_terms_only,
                                              dg_package_field_tags>;
  static_assert(std::is_same_v<tmpl::list<>, ranges_unused>,
                "Received Tags::Range for Tags that are neither arguments to "
                "dg_package_data nor dg_boundary_terms");

  // Fill all fields with random values in [-1,1), then, for each tag with a
  // specified range, overwrite with new random values in [min,max)
  auto fields_on_face =
      make_with_random_values<Variables<face_tags_with_curved_background>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  tmpl::for_each<tmpl::list<RangeTags...>>([&gen, &fields_on_face,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    // If this range tag is for an argument to dg_boundary_terms only, don't try
    // to extract it from this Variables
    if constexpr (tmpl::list_contains_v<face_tags_with_curved_background,
                                        tag>) {
      const std::array<double, 2>& range =
          tuples::get<Tags::Range<tag>>(ranges);
      std::uniform_real_distribution<> local_dist(range[0], range[1]);
      fill_with_random_values(make_not_null(&get<tag>(fields_on_face)),
                              make_not_null(&gen), make_not_null(&local_dist));
    }
  });

  auto unit_normal_covector = make_with_random_values<
      tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>>(
      make_not_null(&gen), make_not_null(&dist), used_for_size);
  tnsr::I<DataVector, FaceDim + 1, Frame::Inertial> unit_normal_vector{};

  if constexpr (not curved_background) {
    const Scalar<DataVector> normal_magnitude = magnitude(unit_normal_covector);
    for (auto& t : unit_normal_covector) {
      t /= get(normal_magnitude);
    }
  } else {
    using inv_spatial_metric = typename detail::inverse_spatial_metric_tag<
        curved_background>::template f<System>;
    detail::adjust_inverse_spatial_metric(
        make_not_null(&get<inv_spatial_metric>(fields_on_face)));
    detail::normalize_vector_and_covector(
        make_not_null(&unit_normal_covector),
        make_not_null(&unit_normal_vector),
        get<inv_spatial_metric>(fields_on_face));
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
  Variables<dg_package_field_tags> package_data{used_for_size.size()};
  if constexpr (curved_background) {
    call_dg_package_data(make_not_null(&package_data), correction,
                         fields_on_face, volume_data, unit_normal_covector,
                         unit_normal_vector, mesh_velocity,
                         tmpl::list<FaceTags...>{});
  } else {
    call_dg_package_data(make_not_null(&package_data), correction,
                         fields_on_face, volume_data, unit_normal_covector,
                         mesh_velocity, tmpl::list<FaceTags...>{});
  }

  // Call python implementation of dg_package_data
  size_t function_name_index = 0;
  tmpl::for_each<dg_package_field_tags>(
      [epsilon, &fields_on_face, &function_name_index, &mesh_velocity,
       &normal_dot_mesh_velocity, &package_data,
       &python_dg_package_data_functions, &python_module, &unit_normal_covector,
       &unit_normal_vector, &volume_data](auto package_data_tag_v) {
        // avoid compiler warnings if there isn't any volume data
        (void)volume_data;
        INFO("Testing package data");
        using package_data_tag = tmpl::type_from<decltype(package_data_tag_v)>;
        using ResultType = typename package_data_tag::type;
        const std::string tag_name = db::tag_name<package_data_tag>();
        CAPTURE(tag_name);
        try {
          CAPTURE(python_module);
          CAPTURE(
              gsl::at(python_dg_package_data_functions, function_name_index));
          CAPTURE(pretty_type::short_name<ResultType>());
          if constexpr (curved_background) {
            const auto python_result =
                pypp::call<ResultType, ConversionClassList>(
                    python_module,
                    gsl::at(python_dg_package_data_functions,
                            function_name_index),
                    get<FaceTags>(fields_on_face)..., unit_normal_covector,
                    unit_normal_vector, mesh_velocity, normal_dot_mesh_velocity,
                    get<VolumeTags>(volume_data)...);
            CHECK_ITERABLE_CUSTOM_APPROX(
                get<package_data_tag>(package_data), python_result,
                Approx::custom().epsilon(epsilon).scale(1.0));
          } else {
            (void)unit_normal_vector;
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
          }
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
  // Fill all fields with random values in [-1,1), then, for each tag with a
  // specified range, overwrite with new random values in [min,max)
  auto interior_package_data =
      make_with_random_values<Variables<dg_package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  tmpl::for_each<tmpl::list<RangeTags...>>([&gen, &interior_package_data,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    // If this range tag is for an argument to dg_package_data only, don't try
    // to extract it from this Variables
    if constexpr (tmpl::list_contains_v<dg_package_field_tags, tag>) {
      const std::array<double, 2>& range =
          tuples::get<Tags::Range<tag>>(ranges);
      std::uniform_real_distribution<> local_dist(range[0], range[1]);
      fill_with_random_values(make_not_null(&get<tag>(interior_package_data)),
                              make_not_null(&gen), make_not_null(&local_dist));
    }
  });

  // Same as above but for exterior data
  auto exterior_package_data =
      make_with_random_values<Variables<dg_package_field_tags>>(
          make_not_null(&gen), make_not_null(&dist), used_for_size);
  tmpl::for_each<tmpl::list<RangeTags...>>([&gen, &exterior_package_data,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    if constexpr (tmpl::list_contains_v<dg_package_field_tags, tag>) {
      const std::array<double, 2>& range =
          tuples::get<Tags::Range<tag>>(ranges);
      std::uniform_real_distribution<> local_dist(range[0], range[1]);
      fill_with_random_values(make_not_null(&get<tag>(exterior_package_data)),
                              make_not_null(&gen), make_not_null(&local_dist));
    }
  });

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
 * - `ranges` is a `TaggedTuple` of
 *   `TestHelpers::evolution::dg::Tags::Range<tag>` specifying a custom range in
 *   which to generate the random values. This can be used for ensuring that
 *   positive quantities are randomly generated on the interval
 *   `[lower_bound,upper_bound)`, choosing `lower_bound` to be `0` or some small
 *   number. The default interval if a tag is not listed is `[-1,1)`. The range
 *   is used for setting the random inputs to `dg_package_data` and
 *   `dg_boundary_terms`.
 */
template <typename System, typename ConversionClassList = tmpl::list<>,
          typename BoundaryCorrection, size_t FaceDim, typename... VolumeTags,
          typename... RangeTags>
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
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
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
      detail::test_with_python<System, ConversionClassList, variables_tags>(
          python_module, python_dg_package_data_functions,
          python_dg_boundary_terms_functions, correction, face_mesh,
          volume_data, ranges, use_moving_mesh, dg_formulation, epsilon,
          tmpl::append<variables_tags, flux_tags, package_temporary_tags,
                       package_primitive_tags>{},
          typename BoundaryCorrection::dg_package_field_tags{});
    }
  }
}
}  // namespace TestHelpers::evolution::dg
