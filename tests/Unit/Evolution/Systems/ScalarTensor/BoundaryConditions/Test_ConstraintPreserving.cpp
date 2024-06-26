// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <random>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/BoundaryConditions/ConstraintPreservingSphericalRadiation.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/BoundaryCondition.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryConditions/ConstraintPreserving.hpp"
#include "Evolution/Systems/ScalarTensor/BoundaryCorrections/ProductOfCorrections.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Evolution/Systems/ScalarTensor/Tags.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/ScalarTensor/KerrSphericalHarmonic.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        ScalarTensor::BoundaryConditions::BoundaryCondition,
        tmpl::list<ScalarTensor::BoundaryConditions::ConstraintPreserving>>>;
  };
};

template <typename U>
void test_dg(const gsl::not_null<std::mt19937*> generator,
             const U& boundary_condition) {
  const size_t num_points = 5;

  std::uniform_real_distribution<> dist(0.1, 1.0);

  const gh::Solutions::WrappedGr<
      ScalarTensor::AnalyticData::KerrSphericalHarmonic>
      analytic_data{1.0,    std::array<double, 3>{{0.0, 0.0, 0.0}},
                    1.0e-5, 30.0,
                    5.0,    std::pair<size_t, int>{0, 0}};

  const auto interior_gamma1 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist), num_points);
  const auto interior_gamma2 = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist), num_points);

  const auto interior_gamma1_scalar =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist), num_points);
  const auto interior_gamma2_scalar =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist), num_points);

  const auto coords =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
          generator, make_not_null(&dist), num_points);

  using Vars = Variables<tmpl::append<
      ScalarTensor::System::variables_tag::tags_list,
      db::wrap_tags_in<::Tags::Flux, ScalarTensor::System::flux_variables,
                       tmpl::size_t<3_st>, Frame::Inertial>,
      tmpl::list<gh::ConstraintDamping::Tags::ConstraintGamma1,
                 gh::ConstraintDamping::Tags::ConstraintGamma2,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 CurvedScalarWave::Tags::ConstraintGamma1,
                 CurvedScalarWave::Tags::ConstraintGamma2>>>;
  using PrimVars = Variables<tmpl::list<>>;

  Vars vars{num_points};
  Vars expected_vars;
  PrimVars prim_vars;

  std::tie(expected_vars, prim_vars) = [&analytic_data, &coords,
                                        &interior_gamma1, &interior_gamma2,
                                        &interior_gamma1_scalar,
                                        &interior_gamma2_scalar]() {
    Vars expected{num_points};
    auto& [spacetime_metric, pi, phi, psi_scalar, pi_scalar, phi_scalar, gamma1,
           gamma2, lapse, shift, inverse_spatial_metric, gamma1_scalar,
           gamma2_scalar] = expected;

    gamma1 = interior_gamma1;
    gamma2 = interior_gamma2;

    gamma1_scalar = interior_gamma1_scalar;
    gamma2_scalar = interior_gamma2_scalar;

    PrimVars local_prim_vars{num_points};

    using tags =
        tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                   gr::Tags::InverseSpatialMetric<DataVector, 3>,
                   gr::Tags::SqrtDetSpatialMetric<DataVector>,
                   gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                   gr::Tags::SpacetimeMetric<DataVector, 3>,
                   ::gh::Tags::Pi<DataVector, 3>,
                   ::gh::Tags::Phi<DataVector, 3>, CurvedScalarWave::Tags::Psi,
                   CurvedScalarWave::Tags::Pi, CurvedScalarWave::Tags::Phi<3>>;

    tuples::tagged_tuple_from_typelist<tags> analytic_vars{};

    analytic_vars = analytic_data.variables(coords, tags{});

    spacetime_metric =
        get<gr::Tags::SpacetimeMetric<DataVector, 3>>(analytic_vars);
    pi = get<::gh::Tags::Pi<DataVector, 3>>(analytic_vars);
    phi = get<::gh::Tags::Phi<DataVector, 3>>(analytic_vars);

    psi_scalar = get<CurvedScalarWave::Tags::Psi>(analytic_vars);
    pi_scalar = get<CurvedScalarWave::Tags::Pi>(analytic_vars);
    phi_scalar = get<CurvedScalarWave::Tags::Phi<3>>(analytic_vars);

    lapse = get<gr::Tags::Lapse<DataVector>>(analytic_vars);
    shift = get<gr::Tags::Shift<DataVector, 3>>(analytic_vars);
    inverse_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(analytic_vars);

    return std::tuple(expected, local_prim_vars);
  }();

  // Pick random direction normal covector, then normalize and compute normal
  // vector.
  tnsr::i<DataVector, 3> normal_covector{num_points};
  get<0>(normal_covector) = 0.5;
  get<1>(normal_covector) = 0.0;
  get<2>(normal_covector) = 0.5;
  const auto magnitude_normal = magnitude(
      normal_covector,
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(expected_vars));
  for (size_t i = 0; i < 3; ++i) {
    normal_covector.get(i) /= get(magnitude_normal);
  }
  const auto normal_vector =
      tenex::evaluate<ti::I>(normal_covector(ti::j) *
                             get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                                 expected_vars)(ti::I, ti::J));

  auto& [spacetime_metric, pi, phi, psi_scalar, pi_scalar, phi_scalar, gamma1,
         gamma2, lapse, shift, inverse_spatial_metric, gamma1_scalar,
         gamma2_scalar] = vars;

  CHECK(
      not boundary_condition
              .dg_ghost(
                  make_not_null(&spacetime_metric), make_not_null(&pi),
                  make_not_null(&phi),

                  make_not_null(&psi_scalar), make_not_null(&pi_scalar),
                  make_not_null(&phi_scalar),

                  make_not_null(&gamma1), make_not_null(&gamma2),
                  make_not_null(&lapse), make_not_null(&shift),

                  make_not_null(&gamma1_scalar), make_not_null(&gamma2_scalar),

                  make_not_null(&inverse_spatial_metric), {}, normal_covector,
                  normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  get<CurvedScalarWave::Tags::Psi>(expected_vars),
                  get<CurvedScalarWave::Tags::Pi>(expected_vars),
                  get<CurvedScalarWave::Tags::Phi<3>>(expected_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                      expected_vars),

                  {}, {}, {}, {}, {},

                  interior_gamma1_scalar, interior_gamma2_scalar,

                  {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})
              .has_value());

  const auto inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;
  const auto spacetime_normal_vector = gr::spacetime_normal_vector(
      get<gr::Tags::Lapse<DataVector>>(expected_vars),
      get<gr::Tags::Shift<DataVector, 3>>(expected_vars));
  const auto gauge_source = make_with_random_values<tnsr::a<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto spacetime_deriv_gauge_source =
      make_with_random_values<tnsr::ab<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto d_spacetime_metric =
      make_with_random_values<tnsr::iaa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto d_pi = make_with_random_values<tnsr::iaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto d_phi = make_with_random_values<tnsr::ijaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);

  const auto d_psi_scalar = make_with_random_values<tnsr::i<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto d_pi_scalar = make_with_random_values<tnsr::i<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto d_phi_scalar = make_with_random_values<tnsr::ij<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);

  const auto three_index_constraint =
      make_with_random_values<tnsr::iaa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto logical_dt_spacetime_metric =
      make_with_random_values<tnsr::aa<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);
  const auto logical_dt_pi = make_with_random_values<tnsr::aa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);
  const auto logical_dt_phi = make_with_random_values<tnsr::iaa<DataVector, 3>>(
      generator, make_not_null(&dist), num_points);

  const auto logical_dt_psi_scalar =
      make_with_random_values<Scalar<DataVector>>(
          generator, make_not_null(&dist), num_points);
  const auto logical_dt_pi_scalar = make_with_random_values<Scalar<DataVector>>(
      generator, make_not_null(&dist), num_points);
  const auto logical_dt_phi_scalar =
      make_with_random_values<tnsr::i<DataVector, 3>>(
          generator, make_not_null(&dist), num_points);

  using DtVars = Variables<db::wrap_tags_in<
      ::Tags::dt, ScalarTensor::System::variables_tag::tags_list>>;
  DtVars dt_vars{num_points};

  // Test CurvedScalarWave constraint-preserving BC
  CHECK(
      not boundary_condition
              .dg_time_derivative(
                  make_not_null(
                      &get<
                          ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
                          dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(dt_vars)),

                  make_not_null(
                      &get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<CurvedScalarWave::Tags::Pi>>(dt_vars)),
                  make_not_null(
                      &get<::Tags::dt<CurvedScalarWave::Tags::Phi<3>>>(
                          dt_vars)),

                  {}, normal_covector, normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  get<CurvedScalarWave::Tags::Psi>(expected_vars),
                  get<CurvedScalarWave::Tags::Pi>(expected_vars),
                  get<CurvedScalarWave::Tags::Phi<3>>(expected_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
                      expected_vars),
                  inverse_spacetime_metric, spacetime_normal_vector,
                  three_index_constraint, gauge_source,
                  spacetime_deriv_gauge_source,

                  interior_gamma1_scalar, interior_gamma2_scalar,

                  logical_dt_spacetime_metric, logical_dt_pi, logical_dt_phi,

                  logical_dt_psi_scalar, logical_dt_pi_scalar,
                  logical_dt_phi_scalar,

                  d_spacetime_metric, d_pi, d_phi,

                  d_psi_scalar, d_pi_scalar, d_phi_scalar)
              .has_value());

  CurvedScalarWave::BoundaryConditions::ConstraintPreservingSphericalRadiation<
      3>
      csw_constraint_preserving{};

  DtVars expected_dt_vars{num_points, 0.0};

  CHECK(not csw_constraint_preserving
                .dg_time_derivative(

                    make_not_null(&get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(
                        expected_dt_vars)),
                    make_not_null(&get<::Tags::dt<CurvedScalarWave::Tags::Pi>>(
                        expected_dt_vars)),
                    make_not_null(
                        &get<::Tags::dt<CurvedScalarWave::Tags::Phi<3>>>(
                            expected_dt_vars)),

                    {}, normal_covector, normal_vector,

                    get<CurvedScalarWave::Tags::Psi>(expected_vars),
                    get<CurvedScalarWave::Tags::Phi<3>>(expected_vars),

                    coords, interior_gamma1_scalar, interior_gamma2_scalar,

                    get<gr::Tags::Lapse<DataVector>>(expected_vars),
                    get<gr::Tags::Shift<DataVector, 3>>(expected_vars),

                    logical_dt_psi_scalar, logical_dt_pi_scalar,
                    logical_dt_phi_scalar, d_psi_scalar, d_pi_scalar,
                    d_phi_scalar)
                .has_value());

  tmpl::for_each<typename Vars::tags_list>([&expected_vars, &vars](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CAPTURE(db::tag_name<tag>());
    CHECK(get<tag>(vars) == get<tag>(expected_vars));
  });

  // Test Generalized Harmonic constraint-preserving BC
  gh::BoundaryConditions::ConstraintPreservingBjorhus<3> gh_cp{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreservingPhysical};
  CHECK(
      not gh_cp
              .dg_time_derivative(
                  make_not_null(
                      &get<
                          ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
                          expected_dt_vars)),
                  make_not_null(&get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(
                      expected_dt_vars)),
                  make_not_null(&get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(
                      expected_dt_vars)),

                  {}, normal_covector, normal_vector,

                  get<gr::Tags::SpacetimeMetric<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Pi<DataVector, 3>>(expected_vars),
                  get<gh::Tags::Phi<DataVector, 3>>(expected_vars),

                  coords, interior_gamma1, interior_gamma2,
                  get<gr::Tags::Lapse<DataVector>>(expected_vars),
                  get<gr::Tags::Shift<DataVector, 3>>(expected_vars),
                  inverse_spacetime_metric, spacetime_normal_vector,
                  three_index_constraint, gauge_source,
                  spacetime_deriv_gauge_source, logical_dt_spacetime_metric,
                  logical_dt_pi, logical_dt_phi, d_spacetime_metric, d_pi,
                  d_phi)
              .has_value());
  tmpl::for_each<typename DtVars::tags_list>(
      [&dt_vars, &expected_dt_vars](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CAPTURE(db::tag_name<tag>());
        CHECK(get<tag>(dt_vars) == get<tag>(expected_dt_vars));
      });
}

}  // namespace

SPECTRE_TEST_CASE(
  "Unit.ScalarTensor.BoundaryConditions.ConstraintPreserving",
  "[Unit][Evolution]") {
  MAKE_GENERATOR(gen);
  register_factory_classes_with_charm<Metavariables>();

  const auto product_boundary_condition =
      TestHelpers::test_creation<
          std::unique_ptr<ScalarTensor::BoundaryConditions::BoundaryCondition>,
          Metavariables>(
          "ConstraintPreserving:\n"
          "  Type: ConstraintPreservingPhysical\n")
          ->get_clone();

  const auto serialized_and_deserialized_condition = serialize_and_deserialize(
      *dynamic_cast<ScalarTensor::BoundaryConditions::ConstraintPreserving*>(
          product_boundary_condition.get()));

  test_dg(make_not_null(&gen), serialized_and_deserialized_condition);
}
