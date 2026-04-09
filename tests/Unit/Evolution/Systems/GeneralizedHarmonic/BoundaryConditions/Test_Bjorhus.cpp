// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Bjorhus.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions/Factory.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryCorrections/UpwindPenalty.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryConditions.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/Sinusoid.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestHelpers::evolution::dg;

namespace {
using frame = Frame::Inertial;

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<MathFunction<1, Frame::Inertial>,
                   tmpl::list<MathFunctions::Sinusoid<1, Frame::Inertial>>>,
        tmpl::pair<gh::BoundaryConditions::BoundaryCondition<Dim>,
                   tmpl::list<gh::BoundaryConditions::
                                  ConstraintPreservingBjorhus<Dim>>>>;
  };
};

template <size_t Dim>
void test() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  for (const auto& [bc_string, bc_type] :
       {std::pair{"ConstraintPreserving"s, "ConstraintPreservingGauge"s},
        std::pair{"ConstraintPreservingPhysical"s,
                  "ConstraintPreservingGaugePhysical"s}}) {
    CAPTURE(bc_string);
    const auto box_of_gridless_data =
        db::create<db::AddSimpleTags<::Tags::Time>>(0.0);

    helpers::test_boundary_condition_with_python<
        gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim>,
        gh::BoundaryConditions::BoundaryCondition<Dim>, gh::System<Dim>,
        tmpl::list<gh::BoundaryCorrections::UpwindPenalty<Dim>>, tmpl::list<>,
        tmpl::list<>, Metavariables<Dim>>(
        make_not_null(&gen),
        "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
        tuples::TaggedTuple<
            helpers::Tags::PythonFunctionForErrorMessage<>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, Dim, frame>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<gh::Tags::Pi<DataVector, Dim, frame>>>,
            helpers::Tags::PythonFunctionName<
                ::Tags::dt<gh::Tags::Phi<DataVector, Dim, frame>>>>{
            "error", "dt_spacetime_metric", "dt_pi_" + bc_type,
            "dt_phi_" + bc_type},
        "ConstraintPreservingBjorhus:\n"
        "  Type: " +
            bc_string +
            (Dim == 3 ? "\n  IncomingWaveProfile: None" : ""),
        Index<Dim - 1>{Dim == 1 ? 1 : 5}, box_of_gridless_data,
        tuples::TaggedTuple<
            helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
            helpers::Tags::Range<gr::Tags::Shift<DataVector, Dim, frame>>,
            helpers::Tags::Range<
                gh::Tags::SpacetimeDerivGaugeH<DataVector, Dim, frame>>,
            helpers::Tags::Range<
                domain::Tags::Coordinates<Dim, Frame::Inertial>>>{
            std::array<double, 2>{{0.8, 1.}}, std::array<double, 2>{{0.1, 0.2}},
            std::array<double, 2>{{0.1, 1.}},
            std::array<double, 2>{{-1000., 1000.}}},
        1.e-6);
  }
}

void test_incoming_wave_profile_option_parsing_and_dim_guard() {
  {
    const auto created = TestHelpers::test_creation<
        std::unique_ptr<gh::BoundaryConditions::BoundaryCondition<3>>,
        Metavariables<3>>(
        "ConstraintPreservingBjorhus:\n"
        "  Type: ConstraintPreservingPhysical\n"
        "  IncomingWaveProfile: None");
    CHECK(dynamic_cast<
              const gh::BoundaryConditions::ConstraintPreservingBjorhus<3>*>(
              created.get()) != nullptr);
  }

  {
    const auto created = TestHelpers::test_creation<
        std::unique_ptr<gh::BoundaryConditions::BoundaryCondition<3>>,
        Metavariables<3>>(
        "ConstraintPreservingBjorhus:\n"
        "  Type: ConstraintPreservingPhysical\n"
        "  IncomingWaveProfile:\n"
        "    Sinusoid:\n"
        "      Amplitude: 1.2\n"
        "      Wavenumber: 0.7\n"
        "      Phase: 0.4");
    CHECK(dynamic_cast<
              const gh::BoundaryConditions::ConstraintPreservingBjorhus<3>*>(
              created.get()) != nullptr);
  }

  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<gh::BoundaryConditions::BoundaryCondition<1>>,
          Metavariables<1>>("ConstraintPreservingBjorhus:\n"
                            "  Type: ConstraintPreservingPhysical\n"
                            "  IncomingWaveProfile:\n"
                            "    Sinusoid:\n"
                            "      Amplitude: 1.2\n"
                            "      Wavenumber: 0.7\n"
                            "      Phase: 0.4")),
      Catch::Matchers::ContainsSubstring(
          "Option 'IncomingWaveProfile' is not a valid option."));

  CHECK_THROWS_WITH(
      (TestHelpers::test_creation<
          std::unique_ptr<gh::BoundaryConditions::BoundaryCondition<2>>,
          Metavariables<2>>("ConstraintPreservingBjorhus:\n"
                            "  Type: ConstraintPreservingPhysical\n"
                            "  IncomingWaveProfile:\n"
                            "    Sinusoid:\n"
                            "      Amplitude: 1.2\n"
                            "      Wavenumber: 0.7\n"
                            "      Phase: 0.4")),
      Catch::Matchers::ContainsSubstring(
          "Option 'IncomingWaveProfile' is not a valid option."));
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreservingGauge(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::I<DataVector, Dim, frame>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) {
  gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim> bjorhus_obj{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreserving};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreservingGauge_static_mesh(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) {
  gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim> bjorhus_obj{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreserving};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      std::nullopt, normal_covector, normal_vector, spacetime_metric, pi, phi,
      coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreservingGaugePhysical(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::I<DataVector, Dim, frame>& face_mesh_velocity,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) {
  gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim> bjorhus_obj{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreservingPhysical};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      face_mesh_velocity, normal_covector, normal_vector, spacetime_metric, pi,
      phi, coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void wrap_dt_vars_corrections_ConstraintPreservingGaugePhysical_static_mesh(
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*>
        dt_spacetime_metric_correction,
    const gsl::not_null<tnsr::aa<DataVector, Dim, frame>*> dt_pi_correction,
    const gsl::not_null<tnsr::iaa<DataVector, Dim, frame>*> dt_phi_correction,
    const tnsr::i<DataVector, Dim, frame>& normal_covector,
    const tnsr::I<DataVector, Dim, frame>& normal_vector,
    // c.f. dg_interior_evolved_variables_tags
    const tnsr::aa<DataVector, Dim, frame>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& pi,
    const tnsr::iaa<DataVector, Dim, frame>& phi,
    // c.f. dg_interior_temporary_tags
    const tnsr::I<DataVector, Dim, frame>& coords,
    const Scalar<DataVector>& gamma1, const Scalar<DataVector>& gamma2,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::AA<DataVector, Dim, Frame::Inertial>& inverse_spacetime_metric,
    const tnsr::A<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_vector,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& three_index_constraint,
    const tnsr::a<DataVector, Dim, frame>& gauge_source,
    const tnsr::ab<DataVector, Dim, frame>& spacetime_deriv_gauge_source,
    // c.f. dg_interior_dt_vars_tags
    const tnsr::aa<DataVector, Dim, frame>& dt_spacetime_metric,
    const tnsr::aa<DataVector, Dim, frame>& dt_pi,
    const tnsr::iaa<DataVector, Dim, frame>& dt_phi,
    // c.f. dg_interior_deriv_vars_tags
    const tnsr::iaa<DataVector, Dim, frame>& d_spacetime_metric,
    const tnsr::iaa<DataVector, Dim, frame>& d_pi,
    const tnsr::ijaa<DataVector, Dim, frame>& d_phi) {
  gh::BoundaryConditions::ConstraintPreservingBjorhus<Dim> bjorhus_obj{
      gh::BoundaryConditions::detail::ConstraintPreservingBjorhusType::
          ConstraintPreservingPhysical};
  bjorhus_obj.dg_time_derivative(
      dt_spacetime_metric_correction, dt_pi_correction, dt_phi_correction,
      std::nullopt, normal_covector, normal_vector, spacetime_metric, pi, phi,
      coords, gamma1, gamma2, lapse, shift, inverse_spacetime_metric,
      spacetime_unit_normal_vector, three_index_constraint, gauge_source,
      spacetime_deriv_gauge_source, dt_spacetime_metric, dt_pi, dt_phi,
      d_spacetime_metric, d_pi, d_phi);
}

template <size_t Dim>
void test_with_random_values(const DataVector& used_for_size) {
  // Static mesh
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreservingGauge_static_mesh<Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric_static_mesh",
       "dt_pi_ConstraintPreservingGauge_static_mesh",
       "dt_phi_ConstraintPreservingGauge_static_mesh"},
      {{{0.1, 1.}}}, used_for_size, 1.e-6);
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreservingGaugePhysical_static_mesh<
          Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric_static_mesh",
       "dt_pi_ConstraintPreservingGaugePhysical_static_mesh",
       "dt_phi_ConstraintPreservingGaugePhysical_static_mesh"},
      {{{0.1, 1.}}}, used_for_size, 1.e-6);

  // Moving mesh
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreservingGauge<Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric", "dt_pi_ConstraintPreservingGauge",
       "dt_phi_ConstraintPreservingGauge"},
      {{{0.1, 1.}}}, used_for_size, 1.e-6);
  pypp::check_with_random_values<1>(
      wrap_dt_vars_corrections_ConstraintPreservingGaugePhysical<Dim>,
      "Evolution.Systems.GeneralizedHarmonic.BoundaryConditions.Bjorhus",
      {"dt_spacetime_metric", "dt_pi_ConstraintPreservingGaugePhysical",
       "dt_phi_ConstraintPreservingGaugePhysical"},
      {{{0.1, 1.}}}, used_for_size, 1.e-6);
}
}  // namespace

// [[TimeOut, 20]]
SPECTRE_TEST_CASE("Unit.Evolution.Systems.GeneralizedHarmonic.BCBjorhus.Cls",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{""};

  test<1>();
  test<2>();
  test<3>();

  const DataVector used_for_size(3);

  test_with_random_values<1>(used_for_size);
  test_with_random_values<2>(used_for_size);
  test_with_random_values<3>(used_for_size);
  test_incoming_wave_profile_option_parsing_and_dim_guard();
}
