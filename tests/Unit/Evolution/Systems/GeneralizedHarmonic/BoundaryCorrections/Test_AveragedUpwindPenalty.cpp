// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <optional>
#include <random>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/AveragedUpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ConstraintDampingTags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Dim>
using gh_tags =
    tmpl::list<gr::Tags::SpacetimeMetric<DataVector, Dim>,
               gh::Tags::Pi<DataVector, Dim>, gh::Tags::Phi<DataVector, Dim>>;
template <size_t Dim>
using GhVars = Variables<gh_tags<Dim>>;

template <size_t Dim>
struct MeshVelocity : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
using Args = Variables<tmpl::push_back<
    gh_tags<Dim>, gh::Tags::ConstraintGamma1, gh::Tags::ConstraintGamma2,
    domain::Tags::UnnormalizedFaceNormal<Dim>, MeshVelocity<Dim>>>;

template <size_t Dim, typename Correction>
Variables<typename Correction::dg_package_field_tags> call_dg_package_data(
    const Correction& correction, const Args<Dim>& args,
    const bool include_mesh_velocity) {
  const Scalar<DataVector> unused_lapse{};
  const tnsr::I<DataVector, Dim> unused_shift{};
  const tnsr::I<DataVector, Dim> unused_normal_vector{};
  const Scalar<DataVector> unused_normal_dot_mesh_velocity{};

  Variables<typename Correction::dg_package_field_tags> packaged_data{
      args.number_of_grid_points()};
  tmpl::as_pack<typename Correction::dg_package_field_tags>(
      [&]<typename... DgPackageFieldTags>(
          tmpl::type_<DgPackageFieldTags>... /*meta*/) {
        correction.dg_package_data(
            make_not_null(&get<DgPackageFieldTags>(packaged_data))...,
            get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(args),
            get<gh::Tags::Pi<DataVector, Dim>>(args),
            get<gh::Tags::Phi<DataVector, Dim>>(args),
            get<gh::Tags::ConstraintGamma1>(args),
            get<gh::Tags::ConstraintGamma2>(args), unused_lapse, unused_shift,
            get<domain::Tags::UnnormalizedFaceNormal<Dim>>(args),
            unused_normal_vector,
            include_mesh_velocity ? std::optional{get<MeshVelocity<Dim>>(args)}
                                  : std::nullopt,
            include_mesh_velocity
                ? std::optional{unused_normal_dot_mesh_velocity}
                : std::nullopt);
      });
  return packaged_data;
}

template <size_t Dim, typename Correction>
GhVars<Dim> call_dg_boundary_terms(
    const Correction& correction,
    const Variables<typename Correction::dg_package_field_tags>& packaged_int,
    const Variables<typename Correction::dg_package_field_tags>& packaged_ext) {
  GhVars<Dim> boundary_terms{packaged_int.number_of_grid_points()};
  tmpl::as_pack<gh_tags<Dim>>(
      [&]<typename... GhVarTags>(tmpl::type_<GhVarTags>... /*meta*/) {
        tmpl::as_pack<typename Correction::dg_package_field_tags>(
            [&]<typename... DgPackageFieldTags>(
                tmpl::type_<DgPackageFieldTags>... /*meta*/) {
              correction.dg_boundary_terms(
                  make_not_null(&get<GhVarTags>(boundary_terms))...,
                  get<DgPackageFieldTags>(packaged_int)...,
                  get<DgPackageFieldTags>(packaged_ext)...,
                  dg::Formulation::StrongInertial);
            });
      });
  return boundary_terms;
}

// Note: The exterior normal is negated before calling the correction
template <size_t Dim>
GhVars<Dim> boundary_correction(const Args<Dim>& interior, Args<Dim> exterior,
                                const bool include_mesh_velocity = true) {
  const gh::BoundaryCorrections::AveragedUpwindPenalty<Dim> correction{};

  for (auto& component :
       get<domain::Tags::UnnormalizedFaceNormal<Dim>>(exterior)) {
    component *= -1.0;
  }

  const auto packaged_int =
      call_dg_package_data<Dim>(correction, interior, include_mesh_velocity);
  const auto packaged_ext =
      call_dg_package_data<Dim>(correction, exterior, include_mesh_velocity);
  return call_dg_boundary_terms<Dim>(correction, packaged_int, packaged_ext);
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> gen, const size_t num_pts) {
  register_classes_with_charm<
      gh::BoundaryCorrections::AveragedUpwindPenalty<Dim>>();

  CHECK(gh::BoundaryCorrections::AveragedUpwindPenalty<Dim>{} ==
        gh::BoundaryCorrections::AveragedUpwindPenalty<Dim>{});
  CHECK_FALSE(gh::BoundaryCorrections::AveragedUpwindPenalty<Dim>{} !=
              gh::BoundaryCorrections::AveragedUpwindPenalty<Dim>{});

  std::uniform_real_distribution dist(-0.1, 0.1);
  std::uniform_real_distribution velocity_dist(-2.0, 2.0);
  auto average_fields =
      make_with_random_values<Args<Dim>>(gen, make_not_null(&dist), num_pts);
  // Use a larger range for the mesh velocity so we get all sign
  // combinations for the speeds.
  fill_with_random_values(
      make_not_null(&get<MeshVelocity<Dim>>(average_fields)), gen,
      make_not_null(&velocity_dist));

  // Make the metric reasonable
  {
    auto& metric =
        get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(average_fields);
    get<0, 0>(metric) -= 1.0;
    for (size_t i = 1; i <= Dim; ++i) {
      metric.get(i, i) += 1.0;
    }
  }

  const auto delta_fields =
      make_with_random_values<Args<Dim>>(gen, make_not_null(&dist), num_pts);
  const auto correction = boundary_correction<Dim>(
      average_fields - delta_fields, average_fields + delta_fields);

  // Check that the function at one point matches a single point
  // calculation
  {
    const auto single_point_average =
        extract_point(average_fields, num_pts / 2);
    const auto single_point_delta = extract_point(delta_fields, num_pts / 2);
    const auto single_point_correction =
        boundary_correction<Dim>(single_point_average - single_point_delta,
                                 single_point_average + single_point_delta);
    CHECK_VARIABLES_APPROX(single_point_correction,
                           extract_point(correction, num_pts / 2));
  }

  // Check that the solution is linear in the jump if the average is
  // kept constant
  {
    const auto correction_minus4 =
        boundary_correction<Dim>(average_fields + 4.0 * delta_fields,
                                 average_fields - 4.0 * delta_fields);
    CHECK_VARIABLES_APPROX(correction_minus4, -4.0 * correction);
  }

  // Check that zero mesh velocity gives the same result as a fixed mesh
  {
    auto zero_mesh_average = average_fields;
    auto zero_mesh_delta = delta_fields;
    for (auto& component : get<MeshVelocity<Dim>>(zero_mesh_average)) {
      component = 0.0;
    }
    for (auto& component : get<MeshVelocity<Dim>>(zero_mesh_delta)) {
      component = 0.0;
    }

    const auto zero_mesh_correction =
        boundary_correction<Dim>(zero_mesh_average - zero_mesh_delta,
                                 zero_mesh_average + zero_mesh_delta);
    const auto fixed_mesh_correction = boundary_correction<Dim>(
        average_fields - delta_fields, average_fields + delta_fields, false);
    CHECK(zero_mesh_correction == fixed_mesh_correction);
  }

  // Actually perform the characteristic decomposition and compare to
  // the result.
  {
    const auto& spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(average_fields);
    const auto spatial_metric = gr::spatial_metric(spacetime_metric);
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;
    const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
    const auto lapse = gr::lapse(shift, spacetime_metric);
    // The evolved<->characteristic functions require a unit normal,
    // but the boundary correction should do the normalization itself
    // (since averaging the two sides doesn't preserve normalization).
    auto unit_normal_one_form =
        get<domain::Tags::UnnormalizedFaceNormal<Dim>>(average_fields);
    const auto normal_magnitude =
        magnitude(unit_normal_one_form, inverse_spatial_metric);
    for (auto& component : unit_normal_one_form) {
      component /= get(normal_magnitude);
    }

    auto speeds = gh::characteristic_speeds(
        get<gh::Tags::ConstraintGamma1>(average_fields), lapse, shift,
        unit_normal_one_form,
        std::optional{get<MeshVelocity<Dim>>(average_fields)});

    using CharacteristicFields = Variables<
        tmpl::list<gh::Tags::VSpacetimeMetric<DataVector, Dim, Frame::Inertial>,
                   gh::Tags::VZero<DataVector, Dim, Frame::Inertial>,
                   gh::Tags::VPlus<DataVector, Dim, Frame::Inertial>,
                   gh::Tags::VMinus<DataVector, Dim, Frame::Inertial>>>;

    // Factor of 2.0 comes from (avg + delta) - (avg - delta) = 2 delta
    CharacteristicFields characteristic_correction =
        2.0 * gh::characteristic_fields(
                  get<gh::Tags::ConstraintGamma2>(average_fields),
                  inverse_spatial_metric,
                  get<gr::Tags::SpacetimeMetric<DataVector, Dim>>(delta_fields),
                  get<gh::Tags::Pi<DataVector, Dim>>(delta_fields),
                  get<gh::Tags::Phi<DataVector, Dim>>(delta_fields),
                  unit_normal_one_form);
    for (size_t p = 0; p < num_pts; ++p) {
      for (auto& component :
           get<gh::Tags::VSpacetimeMetric<DataVector, Dim, Frame::Inertial>>(
               characteristic_correction)) {
        component[p] *= speeds[0][p] < 0.0 ? speeds[0][p] : 0.0;
      }
      for (auto& component :
           get<gh::Tags::VZero<DataVector, Dim, Frame::Inertial>>(
               characteristic_correction)) {
        component[p] *= speeds[1][p] < 0.0 ? speeds[1][p] : 0.0;
      }
      for (auto& component :
           get<gh::Tags::VPlus<DataVector, Dim, Frame::Inertial>>(
               characteristic_correction)) {
        component[p] *= speeds[2][p] < 0.0 ? speeds[2][p] : 0.0;
      }
      for (auto& component :
           get<gh::Tags::VMinus<DataVector, Dim, Frame::Inertial>>(
               characteristic_correction)) {
        component[p] *= speeds[3][p] < 0.0 ? speeds[3][p] : 0.0;
      }
    }

    const auto expected_correction =
        gh::evolved_fields_from_characteristic_fields(
            get<gh::Tags::ConstraintGamma2>(average_fields),
            get<gh::Tags::VSpacetimeMetric<DataVector, Dim, Frame::Inertial>>(
                characteristic_correction),
            get<gh::Tags::VZero<DataVector, Dim, Frame::Inertial>>(
                characteristic_correction),
            get<gh::Tags::VPlus<DataVector, Dim, Frame::Inertial>>(
                characteristic_correction),
            get<gh::Tags::VMinus<DataVector, Dim, Frame::Inertial>>(
                characteristic_correction),
            unit_normal_one_form);

    CHECK_VARIABLES_APPROX(correction, expected_correction);
  }
}

SPECTRE_TEST_CASE(
    "Unit.GeneralizedHarmonic.BoundaryCorrections.AveragedUpwindPenalty",
    "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);

  test<1>(make_not_null(&gen), 1);
  test<2>(make_not_null(&gen), 5);
  test<3>(make_not_null(&gen), 5);
}
}  // namespace
