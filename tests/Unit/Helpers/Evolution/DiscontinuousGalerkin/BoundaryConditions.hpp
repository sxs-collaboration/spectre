// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/BoundaryConditions/Periodic.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryConditions/Type.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ComputeTimeDerivativeHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/NormalVectors.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Range.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace TestHelpers::evolution::dg {
/// Tags for testing DG code
namespace Tags {
/// Tag for a `TaggedTuple` that holds the name of the python function that
/// gives the result for computing what `Tag` should be.
///
/// If a Tag is only needed for or different for a specific boundary correction,
/// then the `BoundaryCorrection` template parameter must be set.
///
/// The `Tag` template parameter is an input to the `dg_package_data` function.
/// That is, the tag is one of the evolved variables, fluxes, or any other
/// `dg_package_data_temporary_tags` that the boundary correction needs.
template <typename Tag, typename BoundaryCorrection = NoSuchType>
struct PythonFunctionName {
  using tag = Tag;
  using boundary_correction = BoundaryCorrection;
  using type = std::string;
};

/// The name of the python function that returns the error message.
///
/// If `BoundaryCorrection` is `NoSuchType` then the same function will be used
/// for all boundary corrections.
///
/// The python function must return `None` if there shouldn't be an error
/// message.
template <typename BoundaryCorrection = NoSuchType>
struct PythonFunctionForErrorMessage {
  using boundary_correction = BoundaryCorrection;
  using type = std::string;
};
}  // namespace Tags

namespace detail {
template <typename ConversionClassList, typename... Args>
static std::optional<std::string> call_for_error_message(
    const std::string& module_name, const std::string& function_name,
    const Args&... t) {
  static_assert(sizeof...(Args) > 0,
                "Call to python which returns a Tensor of DataVectors must "
                "pass at least one argument");
  using ReturnType = std::optional<std::string>;

  PyObject* python_module = PyImport_ImportModule(module_name.c_str());
  if (python_module == nullptr) {
    PyErr_Print();
    throw std::runtime_error{std::string("Could not find python module.\n") +
                             module_name};
  }
  PyObject* func = PyObject_GetAttrString(python_module, function_name.c_str());
  if (func == nullptr or not PyCallable_Check(func)) {
    if (PyErr_Occurred()) {
      PyErr_Print();
    }
    throw std::runtime_error{"Could not find python function in module.\n"};
  }

  const std::array<size_t, sizeof...(Args)> arg_sizes{
      {pypp::detail::ContainerPackAndUnpack<
          Args, ConversionClassList>::get_size(t)...}};
  const size_t npts = *std::max_element(arg_sizes.begin(), arg_sizes.end());
  for (size_t i = 0; i < arg_sizes.size(); ++i) {
    if (gsl::at(arg_sizes, i) != 1 and gsl::at(arg_sizes, i) != npts) {
      ERROR("Each argument must return size 1 or "
            << npts
            << " (the number of points in the DataVector), but argument number "
            << i << " has size " << gsl::at(arg_sizes, i));
    }
  }
  ReturnType result{};

  for (size_t s = 0; s < npts; ++s) {
    PyObject* args = pypp::make_py_tuple(
        pypp::detail::ContainerPackAndUnpack<Args, ConversionClassList>::unpack(
            t, s)...);
    auto ret =
        pypp::detail::call_work<typename pypp::detail::ContainerPackAndUnpack<
            ReturnType, ConversionClassList>::unpacked_container>(python_module,
                                                                  func, args);
    if (ret.has_value()) {
      result = std::move(ret);
      break;
    }
  }
  Py_DECREF(func);           // NOLINT
  Py_DECREF(python_module);  // NOLINT
  return result;
}

template <typename BoundaryCorrection, typename... PythonFunctionNameTags>
const std::string& get_python_error_message_function(
    const tuples::TaggedTuple<PythonFunctionNameTags...>&
        python_boundary_condition_functions) {
  return tuples::get<tmpl::conditional_t<
      tmpl::list_contains_v<tmpl::list<PythonFunctionNameTags...>,
                            Tags::PythonFunctionForErrorMessage<NoSuchType>>,
      Tags::PythonFunctionForErrorMessage<NoSuchType>,
      Tags::PythonFunctionForErrorMessage<BoundaryCorrection>>>(
      python_boundary_condition_functions);
}

template <typename Tag, typename BoundaryCorrection,
          typename... PythonFunctionNameTags>
const std::string& get_python_tag_function(
    const tuples::TaggedTuple<PythonFunctionNameTags...>&
        python_boundary_condition_functions) {
  return tuples::get<tmpl::conditional_t<
      tmpl::list_contains_v<tmpl::list<PythonFunctionNameTags...>,
                            Tags::PythonFunctionName<Tag, NoSuchType>>,
      Tags::PythonFunctionName<Tag, NoSuchType>,
      Tags::PythonFunctionName<Tag, BoundaryCorrection>>>(
      python_boundary_condition_functions);
}

template <typename BoundaryConditionHelper, typename AllTagsOnFaceList,
          typename... TagsFromFace, typename... VolumeArgs>
void apply_boundary_condition_impl(
    BoundaryConditionHelper& boundary_condition_helper,
    const Variables<AllTagsOnFaceList>& fields_on_interior_face,
    tmpl::list<TagsFromFace...> /*meta*/,
    const VolumeArgs&... volume_args) noexcept {
  boundary_condition_helper(get<TagsFromFace>(fields_on_interior_face)...,
                            volume_args...);
}

template <typename System, typename ConversionClassList,
          typename BoundaryCorrection, typename... PythonFunctionNameTags,
          typename BoundaryCondition, size_t FaceDim, typename DbTagsList,
          typename... RangeTags, typename... EvolvedVariablesTags,
          typename... BoundaryCorrectionPackagedDataInputTags,
          typename... BoundaryConditionVolumeTags>
void test_boundary_condition_with_python_impl(
    const gsl::not_null<std::mt19937*> generator,
    const std::string& python_module,
    const tuples::TaggedTuple<PythonFunctionNameTags...>&
        python_boundary_condition_functions,
    const BoundaryCondition& boundary_condition,
    const Index<FaceDim>& face_points,
    const db::DataBox<DbTagsList>& box_of_volume_data,
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
    const bool use_moving_mesh, tmpl::list<EvolvedVariablesTags...> /*meta*/,
    tmpl::list<BoundaryCorrectionPackagedDataInputTags...> /*meta*/,
    tmpl::list<BoundaryConditionVolumeTags...> /*meta*/, const double epsilon) {
  CAPTURE(FaceDim);
  CAPTURE(python_module);
  CAPTURE(face_points);
  CAPTURE(use_moving_mesh);
  CAPTURE(epsilon);
  CAPTURE(pretty_type::short_name<BoundaryCorrection>());
  const size_t number_of_points_on_face = face_points.product();

  using variables_tag = typename System::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using flux_variables = typename System::flux_variables;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;

  constexpr bool uses_ghost =
      BoundaryCondition::bc_type ==
          ::evolution::BoundaryConditions::Type::Ghost or
      BoundaryCondition::bc_type ==
          ::evolution::BoundaryConditions::Type::GhostAndTimeDerivative;
  constexpr bool uses_time_derivative_condition =
      BoundaryCondition::bc_type ==
          ::evolution::BoundaryConditions::Type::TimeDerivative or
      BoundaryCondition::bc_type ==
          ::evolution::BoundaryConditions::Type::GhostAndTimeDerivative;

  // List that holds the inverse spatial metric if it's needed
  using inverse_spatial_metric_list =
      ::evolution::dg::Actions::detail::inverse_spatial_metric_tag<System>;
  constexpr bool has_inv_spatial_metric =
      ::evolution::dg::Actions::detail::has_inverse_spatial_metric_tag_v<
          System>;

  // Set up tags for boundary conditions
  using bcondition_interior_temp_tags =
      typename BoundaryCondition::dg_interior_temporary_tags;
  using bcondition_interior_prim_tags =
      typename ::evolution::dg::Actions::detail::get_primitive_vars<
          System::has_primitive_and_conservative_vars>::
          template boundary_condition_interior_tags<BoundaryCondition>;
  using bcondition_interior_evolved_vars_tags =
      typename BoundaryCondition::dg_interior_evolved_variables_tags;
  using bcondition_interior_dt_evolved_vars_tags =
      ::evolution::dg::Actions::detail::get_dt_vars_from_boundary_condition<
          BoundaryCondition>;
  using bcondition_interior_deriv_evolved_vars_tags =
      ::evolution::dg::Actions::detail::get_deriv_vars_from_boundary_condition<
          BoundaryCondition>;
  using bcondition_interior_tags = tmpl::remove_duplicates<tmpl::append<
      tmpl::conditional_t<has_inv_spatial_metric,
                          tmpl::list<::evolution::dg::Actions::detail::
                                         NormalVector<FaceDim + 1>>,
                          tmpl::list<>>,
      bcondition_interior_evolved_vars_tags, bcondition_interior_prim_tags,
      bcondition_interior_temp_tags, bcondition_interior_dt_evolved_vars_tags,
      bcondition_interior_deriv_evolved_vars_tags>>;

  std::uniform_real_distribution<> dist(-1., 1.);

  // Fill all fields with random values in [-1,1), then, for each tag with a
  // specified range, overwrite with new random values in [min,max)
  Variables<tmpl::remove_duplicates<
      tmpl::append<bcondition_interior_tags, inverse_spatial_metric_list>>>
      interior_face_fields{number_of_points_on_face};
  fill_with_random_values(make_not_null(&interior_face_fields), generator,
                          make_not_null(&dist));
  tmpl::for_each<tmpl::list<RangeTags...>>([&generator, &interior_face_fields,
                                            &ranges](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    const std::array<double, 2>& range = tuples::get<Tags::Range<tag>>(ranges);
    std::uniform_real_distribution<> local_dist(range[0], range[1]);
    fill_with_random_values(make_not_null(&get<tag>(interior_face_fields)),
                            generator, make_not_null(&local_dist));
  });

  auto interior_normal_covector =  // Unnormalized at this point
      make_with_random_values<
          tnsr::i<DataVector, FaceDim + 1, Frame::Inertial>>(
          generator, make_not_null(&dist), number_of_points_on_face);
  if constexpr (has_inv_spatial_metric) {
    auto& inv_spatial_metric =
        get<tmpl::front<inverse_spatial_metric_list>>(interior_face_fields);
    detail::adjust_spatial_metric_or_inverse(
        make_not_null(&inv_spatial_metric));
    auto& normal_vector =
        get<::evolution::dg::Actions::detail::NormalVector<FaceDim + 1>>(
            interior_face_fields);
    detail::normalize_vector_and_covector(
        make_not_null(&interior_normal_covector), make_not_null(&normal_vector),
        inv_spatial_metric);
  } else {
    const Scalar<DataVector> magnitude = ::magnitude(interior_normal_covector);
    for (DataVector& component : interior_normal_covector) {
      component /= get(magnitude);
    }
  }

  // Set a random mesh velocity. We allow for velocities above 1 to test "faster
  // than light" mesh movement.
  std::optional<tnsr::I<DataVector, FaceDim + 1>> face_mesh_velocity{};
  if (use_moving_mesh) {
    std::uniform_real_distribution<> local_dist(-2., 2.);
    face_mesh_velocity =
        make_with_random_values<tnsr::I<DataVector, FaceDim + 1>>(
            generator, make_not_null(&local_dist), number_of_points_on_face);
  }

  if constexpr (BoundaryCondition::bc_type ==
                ::evolution::BoundaryConditions::Type::Outflow) {
    // Outflow boundary conditions only check that all characteristic speeds
    // are directed out of the element. If there are any inward directed
    // fields then the boundary condition should error.
    const auto apply_bc =
        [&boundary_condition, &face_mesh_velocity, &interior_normal_covector,
         &python_boundary_condition_functions,
         &python_module](const auto&... face_and_volume_args) noexcept {
          const std::optional<std::string> error_msg =
              boundary_condition.dg_outflow(face_mesh_velocity,
                                            interior_normal_covector,
                                            face_and_volume_args...);
          const std::string& python_error_msg_function =
              get_python_error_message_function<BoundaryCorrection>(
                  python_boundary_condition_functions);
          const auto python_error_message =
              call_for_error_message<ConversionClassList>(
                  python_module, python_error_msg_function, face_mesh_velocity,
                  interior_normal_covector, face_and_volume_args...);
          CAPTURE(python_error_msg_function);
          CAPTURE(python_error_message.value_or(""));
          CAPTURE(error_msg.value_or(""));
          REQUIRE(python_error_message.has_value() == error_msg.has_value());
          if (python_error_message.has_value() and error_msg.has_value()) {
            std::smatch matcher{};
            CHECK(std::regex_search(*error_msg, matcher,
                                    std::regex{*python_error_message}));
          }
        };
    apply_boundary_condition_impl(
        apply_bc, interior_face_fields, bcondition_interior_tags{},
        db::get<BoundaryConditionVolumeTags>(box_of_volume_data)...);
  }

  if constexpr (uses_time_derivative_condition) {
    Variables<dt_variables_tags> time_derivative_correction{
        number_of_points_on_face};
    auto apply_bc = [&boundary_condition, epsilon, &face_mesh_velocity,
                     &interior_normal_covector,
                     &python_boundary_condition_functions, &python_module,
                     &time_derivative_correction](
                        const auto&... interior_face_and_volume_args) {
      const std::optional<std::string> error_msg =
          boundary_condition.dg_time_derivative(
              make_not_null(&get<::Tags::dt<EvolvedVariablesTags>>(
                  time_derivative_correction))...,
              face_mesh_velocity, interior_normal_covector,
              interior_face_and_volume_args...);

      const std::string& python_error_msg_function =
          get_python_error_message_function<BoundaryCorrection>(
              python_boundary_condition_functions);
      const auto python_error_message =
          call_for_error_message<ConversionClassList>(
              python_module, python_error_msg_function, face_mesh_velocity,
              interior_normal_covector, interior_face_and_volume_args...);
      CAPTURE(python_error_msg_function);
      CAPTURE(python_error_message.value_or(""));
      CAPTURE(error_msg.value_or(""));
      REQUIRE(python_error_message.has_value() == error_msg.has_value());
      if (python_error_message.has_value() and error_msg.has_value()) {
        std::smatch matcher{};
        CHECK(std::regex_search(*error_msg, matcher,
                                std::regex{*python_error_message}));
      }

      // Check that the values were set correctly
      tmpl::for_each<dt_variables_tags>([&](auto dt_var_tag_v) {
        using DtVarTag = tmpl::type_from<decltype(dt_var_tag_v)>;
        // Use NoSuchType for the BoundaryCorrection since dt-type corrections
        // should be boundary correction agnostic.
        const std::string& python_tag_function =
            get_python_tag_function<DtVarTag, NoSuchType>(
                python_boundary_condition_functions);
        CAPTURE(python_tag_function);
        CAPTURE(pretty_type::short_name<DtVarTag>());
        typename DtVarTag::type python_result{};
        try {
          python_result =
              pypp::call<typename DtVarTag::type, ConversionClassList>(
                  python_module, python_tag_function, face_mesh_velocity,
                  interior_normal_covector, interior_face_and_volume_args...);
        } catch (const std::exception& e) {
          INFO("Failed python call with '" << e.what() << "'");
          // Use REQUIRE(false) to print all the CAPTURE variables
          REQUIRE(false);
        }
        CHECK_ITERABLE_CUSTOM_APPROX(
            get<DtVarTag>(time_derivative_correction), python_result,
            Approx::custom().epsilon(epsilon).scale(1.0));
      });
    };
    apply_boundary_condition_impl(
        apply_bc, interior_face_fields, bcondition_interior_tags{},
        db::get<BoundaryConditionVolumeTags>(box_of_volume_data)...);
  }

  if constexpr (uses_ghost) {
    using fluxes_tags =
        db::wrap_tags_in<::Tags::Flux, flux_variables,
                         tmpl::size_t<FaceDim + 1>, Frame::Inertial>;
    using correction_temp_tags =
        typename BoundaryCorrection::dg_package_data_temporary_tags;
    using correction_prim_tags =
        typename ::evolution::dg::Actions::detail::get_primitive_vars<
            System::has_primitive_and_conservative_vars>::
            template f<BoundaryCorrection>;
    using tags_on_exterior_face = tmpl::append<
        variables_tags, fluxes_tags, correction_temp_tags, correction_prim_tags,
        inverse_spatial_metric_list,
        tmpl::list<
            ::evolution::dg::Actions::detail::OneOverNormalVectorMagnitude,
            ::evolution::dg::Actions::detail::NormalVector<FaceDim + 1>,
            ::evolution::dg::Tags::NormalCovector<FaceDim + 1>>>;
    Variables<tags_on_exterior_face> exterior_face_fields{
        number_of_points_on_face};
    auto apply_bc = [&boundary_condition, epsilon, &exterior_face_fields,
                     &face_mesh_velocity, &interior_normal_covector,
                     &python_boundary_condition_functions, &python_module](
                        const auto&... interior_face_and_volume_args) {
      std::optional<std::string> error_msg{};
      if constexpr (has_inv_spatial_metric) {
        error_msg = boundary_condition.dg_ghost(
            make_not_null(&get<BoundaryCorrectionPackagedDataInputTags>(
                exterior_face_fields))...,
            make_not_null(&get<typename System::inverse_spatial_metric_tag>(
                exterior_face_fields)),
            face_mesh_velocity, interior_normal_covector,
            interior_face_and_volume_args...);
      } else {
        error_msg = boundary_condition.dg_ghost(
            make_not_null(&get<BoundaryCorrectionPackagedDataInputTags>(
                exterior_face_fields))...,
            face_mesh_velocity, interior_normal_covector,
            interior_face_and_volume_args...);
      }

      const std::string& python_error_msg_function =
          get_python_error_message_function<BoundaryCorrection>(
              python_boundary_condition_functions);
      const auto python_error_message =
          call_for_error_message<ConversionClassList>(
              python_module, python_error_msg_function, face_mesh_velocity,
              interior_normal_covector, interior_face_and_volume_args...);
      CAPTURE(python_error_msg_function);
      CAPTURE(python_error_message.value_or(""));
      CAPTURE(error_msg.value_or(""));
      REQUIRE(python_error_message.has_value() == error_msg.has_value());
      if (python_error_message.has_value() and error_msg.has_value()) {
        std::smatch matcher{};
        CHECK(std::regex_search(*error_msg, matcher,
                                std::regex{*python_error_message}));
      }
      if (python_error_message.has_value()) {
        return;
      }

      // Check that the values were set correctly
      tmpl::for_each<tmpl::list<BoundaryCorrectionPackagedDataInputTags...>>(
          [&](auto boundary_correction_tag_v) {
            using BoundaryCorrectionTag =
                tmpl::type_from<decltype(boundary_correction_tag_v)>;
            const std::string& python_tag_function =
                get_python_tag_function<BoundaryCorrectionTag,
                                        BoundaryCorrection>(
                    python_boundary_condition_functions);
            CAPTURE(python_tag_function);
            CAPTURE(pretty_type::short_name<BoundaryCorrectionTag>());
            typename BoundaryCorrectionTag::type python_result{};
            try {
              python_result = pypp::call<typename BoundaryCorrectionTag::type,
                                         ConversionClassList>(
                  python_module, python_tag_function, face_mesh_velocity,
                  interior_normal_covector, interior_face_and_volume_args...);
            } catch (const std::exception& e) {
              INFO("Failed python call with '" << e.what() << "'");
              // Use REQUIRE(false) to print all the CAPTURE variables
              REQUIRE(false);
            }
            CHECK_ITERABLE_CUSTOM_APPROX(
                get<BoundaryCorrectionTag>(exterior_face_fields), python_result,
                Approx::custom().epsilon(epsilon).scale(1.0));
          });
    };
    // Since any of the variables in `exterior_face_fields` could be mutated
    // from their projected state, we don't pass in the interior tags explicitly
    // since that will likely give a false sense of "no aliasing"
    apply_boundary_condition_impl(
        apply_bc, interior_face_fields, bcondition_interior_tags{},
        db::get<BoundaryConditionVolumeTags>(box_of_volume_data)...);
  }
}

template <typename T, typename = std::void_t<>>
struct get_boundary_conditions_impl {
  using type = tmpl::list<>;
};

template <typename T>
struct get_boundary_conditions_impl<
    T, std::void_t<typename T::boundary_conditions_base>> {
  using type = typename T::boundary_conditions_base::creatable_classes;
};

template <typename T>
using get_boundary_conditions = typename get_boundary_conditions_impl<T>::type;
}  // namespace detail

/*!
 * \brief Test a boundary condition against python code and that it satisfies
 * the required interface.
 *
 * The boundary conditions return a `std::optional<std::string>` that is the
 * error message. For ghost cell boundary conditions they must also return the
 * arguments needed by the boundary correction's `dg_package_data` function by
 * `gsl::not_null`. Time derivative boundary conditions return the correction
 * added to the time derivatives by `gsl::not_null`, while outflow boundary
 * conditions should only check that the boundary is actually an outflow
 * boundary. Therefore, the comparison implementation in python must have a
 * function for each of these. Which function is called is specified using a
 * `python_boundary_condition_functions` and `Tags::PythonFunctionName`.
 *
 * The specific boundary condition and system need to be explicitly given.
 * Once the boundary condition and boundary correction base classes are
 * listed/available in the `System` class we can remove needing to specify
 * those. The `ConversionClassList` is forwarded to `pypp` to allow custom
 * conversions of classes to python, such as analytic solutions or equations of
 * state.
 *
 * - The random number generator is passed in so that the seed can be easily
 *   controlled externally. Use, e.g. `MAKE_GENERATOR(gen)` to create a
 *   generator.
 * - The python function names are given in a `TaggedTuple`. A
 *   `TestHelpers::evolution::dg::Tags::PythonFunctionForErrorMessage` tag must
 *   be given for specifying the error message that the boundary condition
 *   returns. In many cases this function will simply return `None`. The tag can
 *   optionally specify the boundary correction for which it is to be used, so
 *   that a different error message could be printed if a different boundary
 *   correction is used.
 *   The `TestHelpers::evolution::dg::Tags::PythonFunctionName` tag is used to
 *   give the name of the python function for each return argument. The tags are
 *   the input to the `package_data` function for Ghost boundary conditions, and
 *   `::Tags::dt<evolved_var_tag>` for time derivative boundary conditions.
 * - `factory_string` is a string used to create the boundary condition from the
 *   factory
 * - `face_points` is the grid points on the interface. Generally 5 grid points
 *   per dimension in 2d and 3d is recommended to catch indexing errors. In 1d
 *   there is only ever one point on the interface
 * - `box_of_volume_data` is a `db::DataBox` that contains all of the
 *   `dg_gridless_tags` of the boundary condition. This is not a `TaggedTuple`
 *   so that all the different types of tags, like base tags, can be supported
 *   and easily tested.
 * - `ranges` is a `TaggedTuple` of
 *   `TestHelpers::evolution::dg::Tags::Range<tag>` specifying a custom range in
 *   which to generate the random values. This can be used for ensuring that
 *   positive quantities are randomly generated on the interval
 *   `[lower_bound,upper_bound)`, choosing `lower_bound` to be `0` or some small
 *   number. The default interval if a tag is not listed is `[-1,1)`.
 * - `epsilon` is the relative tolerance to use in the random tests
 */
template <typename BoundaryCondition, typename BoundaryConditionBase,
          typename System, typename BoundaryCorrectionsList,
          typename ConversionClassList = tmpl::list<>, size_t FaceDim,
          typename DbTagsList, typename... RangeTags,
          typename... PythonFunctionNameTags>
void test_boundary_condition_with_python(
    const gsl::not_null<std::mt19937*> generator,
    const std::string& python_module,
    const tuples::TaggedTuple<PythonFunctionNameTags...>&
        python_boundary_condition_functions,
    const std::string& factory_string, const Index<FaceDim>& face_points,
    const db::DataBox<DbTagsList>& box_of_volume_data,
    const tuples::TaggedTuple<Tags::Range<RangeTags>...>& ranges,
    const double epsilon = 1.0e-12) {
  PUPable_reg(BoundaryCondition);
  static_assert(std::is_final_v<std::decay_t<BoundaryCondition>>,
                "All boundary condition classes must be marked `final`.");
  using variables_tags = typename System::variables_tag::tags_list;
  using flux_variables = typename System::flux_variables;
  using fluxes_tags =
      db::wrap_tags_in<::Tags::Flux, flux_variables, tmpl::size_t<FaceDim + 1>,
                       Frame::Inertial>;
  const std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
      boundary_condition =
          TestHelpers::test_factory_creation<BoundaryConditionBase,
                                             BoundaryCondition>(factory_string);

  REQUIRE_FALSE(
      domain::BoundaryConditions::is_periodic(boundary_condition->get_clone()));

  tmpl::for_each<BoundaryCorrectionsList>(
      [&boundary_condition, &box_of_volume_data, epsilon, &face_points,
       &generator, &python_boundary_condition_functions, &python_module,
       &ranges](auto boundary_correction_v) {
        using BoundaryCorrection =
            tmpl::type_from<decltype(boundary_correction_v)>;
        using package_data_input_tags = tmpl::append<
            variables_tags, fluxes_tags,
            typename BoundaryCorrection::dg_package_data_temporary_tags,
            typename ::evolution::dg::Actions::detail::get_primitive_vars<
                System::has_primitive_and_conservative_vars>::
                template f<BoundaryCorrection>>;
        using boundary_condition_dg_gridless_tags =
            typename BoundaryCondition::dg_gridless_tags;
        for (const auto use_moving_mesh : {false, true}) {
          detail::test_boundary_condition_with_python_impl<
              System, ConversionClassList, BoundaryCorrection>(
              generator, python_module, python_boundary_condition_functions,
              dynamic_cast<const BoundaryCondition&>(*boundary_condition),
              face_points, box_of_volume_data, ranges, use_moving_mesh,
              variables_tags{}, package_data_input_tags{},
              boundary_condition_dg_gridless_tags{}, epsilon);
          // Now serialize and deserialize and test again
          INFO("Test boundary condition after serialization.");
          const auto deserialized_bc =
              serialize_and_deserialize(boundary_condition);
          detail::test_boundary_condition_with_python_impl<
              System, ConversionClassList, BoundaryCorrection>(
              generator, python_module, python_boundary_condition_functions,
              dynamic_cast<const BoundaryCondition&>(*deserialized_bc),
              face_points, box_of_volume_data, ranges, use_moving_mesh,
              variables_tags{}, package_data_input_tags{},
              boundary_condition_dg_gridless_tags{}, epsilon);
        }
      });
}

/// Test that a boundary condition is correctly identified as periodic.
template <typename BoundaryCondition, typename BoundaryConditionBase>
void test_periodic_condition(const std::string& factory_string) {
  PUPable_reg(BoundaryCondition);
  const auto boundary_condition =
      TestHelpers::test_factory_creation<BoundaryConditionBase,
                                         BoundaryCondition>(factory_string);
  REQUIRE(typeid(*boundary_condition.get()) == typeid(BoundaryCondition));
  const auto bc_clone = boundary_condition->get_clone();
  CHECK(domain::BoundaryConditions::is_periodic(bc_clone));
  const auto bc_deserialized = serialize_and_deserialize(bc_clone);
  CHECK(domain::BoundaryConditions::is_periodic(bc_deserialized));
}
}  // namespace TestHelpers::evolution::dg
