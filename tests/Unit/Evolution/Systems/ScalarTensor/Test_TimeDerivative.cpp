// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/System.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/TimeDerivative.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Tags/GaugeCondition.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Evolution/Systems/ScalarTensor/System.hpp"
#include "Evolution/Systems/ScalarTensor/TimeDerivative.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarTensor.TimeDerivative",
                  "[Unit][Evolution]") {

  using gh_dt_variables_tags = ScalarTensor::TimeDerivative::gh_dt_tags;
  using scalar_dt_variables_tags = ScalarTensor::TimeDerivative::scalar_dt_tags;
  using dt_variables_type =
      Variables<tmpl::append<gh_dt_variables_tags, scalar_dt_variables_tags>>;

  using temp_variables_type =
      Variables<typename ScalarTensor::TimeDerivative::temporary_tags>;

  using gh_gradient_tags = tmpl::transform<
      ScalarTensor::TimeDerivative::gh_gradient_tags,
      tmpl::bind<::Tags::deriv, tmpl::_1, tmpl::pin<tmpl::size_t<3>>,
                 tmpl::pin<Frame::Inertial>>>;
  using scalar_gradient_tags = tmpl::transform<
      ScalarTensor::TimeDerivative::scalar_gradient_tags,
      tmpl::bind<::Tags::deriv, tmpl::_1, tmpl::pin<tmpl::size_t<3>>,
                 tmpl::pin<Frame::Inertial>>>;
  using gradient_variables_type =
      Variables<tmpl::append<gh_gradient_tags, scalar_gradient_tags>>;

  using gh_arg_tags = ScalarTensor::TimeDerivative::gh_arg_tags;
  using scalar_arg_tags = ScalarTensor::TimeDerivative::scalar_arg_tags;
  using arg_variables_type = tuples::tagged_tuple_from_typelist<
      tmpl::append<gh_arg_tags, scalar_arg_tags,
                   tmpl::list<ScalarTensor::Tags::ScalarSource>>>;

  const size_t element_size = 10;
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(0.1, 1.0);

  dt_variables_type expected_dt_variables{element_size};
  dt_variables_type dt_variables{element_size};

  temp_variables_type expected_temp_variables{element_size};
  temp_variables_type temp_variables{element_size};

  const auto gradient_variables =
      make_with_random_values<gradient_variables_type>(
          make_not_null(&gen), make_not_null(&dist), DataVector{element_size});

  arg_variables_type arg_variables;
  tmpl::for_each<tmpl::append<gh_arg_tags, scalar_arg_tags,
                              tmpl::list<ScalarTensor::Tags::ScalarSource>>>(
      [&gen, &dist, &arg_variables](auto tag_v) {
        using tag = typename decltype(tag_v)::type;
        if constexpr (std::is_same_v<typename tag::type,
                                     std::optional<tnsr::I<DataVector, 3,
                                                           Frame::Inertial>>>) {
          tuples::get<tag>(arg_variables) = make_with_random_values<
              typename tnsr::I<DataVector, 3, Frame::Inertial>>(
              make_not_null(&gen), make_not_null(&dist),
              DataVector{element_size});
        } else if constexpr (tt::is_a_v<Tensor, typename tag::type>) {
          tuples::get<tag>(arg_variables) =
              make_with_random_values<typename tag::type>(
                  make_not_null(&gen), make_not_null(&dist),
                  DataVector{element_size});
        }
      });
  get<gh::gauges::Tags::GaugeCondition>(arg_variables) =
      std::make_unique<gh::gauges::DampedHarmonic>(
          100., std::array{1.2, 1.5, 1.7}, std::array{2, 4, 6});

  // ensure that the signature of the metric is correct
  {
    auto& metric =
        tuples::get<gr::Tags::SpacetimeMetric<DataVector, 3>>(arg_variables);
    get<0, 0>(metric) += -2.0;
    for (size_t i = 0; i < 3; ++i) {
      metric.get(i + 1, i + 1) += 4.0;
      metric.get(i + 1, 0) *= 0.01;
    }
  }

  // The logic of the test is the following:
  // We compute the individual time derivative functions for each system
  // and then compare the results with the time derivative function for the
  // combined system

  // The time derivative function for GeneralizedHarmonic is
  gh::TimeDerivative<3>::apply(
      // GH evolved variables
      make_not_null(&get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          expected_dt_variables)),
      make_not_null(
          &get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(expected_dt_variables)),
      make_not_null(&get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(
          expected_dt_variables)),
      // GH temporaries
      make_not_null(&get<gh::ConstraintDamping::Tags::ConstraintGamma1>(
          expected_temp_variables)),
      make_not_null(&get<gh::ConstraintDamping::Tags::ConstraintGamma2>(
          expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::GaugeH<DataVector, 3>>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::SpacetimeDerivGaugeH<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(&get<gh::Tags::Gamma1Gamma2>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::HalfPiTwoNormals>(expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::NormalDotOneIndexConstraint>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::Gamma1Plus1>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::PiOneNormal<3>>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::GaugeConstraint<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::HalfPhiTwoNormals<3>>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::ShiftDotThreeIndexConstraint<3>>(
          expected_temp_variables)),
      make_not_null(&get<gh::Tags::MeshVelocityDotThreeIndexConstraint<3>>(
          expected_temp_variables)),
      make_not_null(&get<gh::Tags::PhiOneNormal<3>>(expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::PiSecondIndexUp<3>>(expected_temp_variables)),
      make_not_null(&get<gh::Tags::ThreeIndexConstraint<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::PhiFirstIndexUp<3>>(expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::PhiThirdIndexUp<3>>(expected_temp_variables)),
      make_not_null(
          &get<gh::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<3>>(
              expected_temp_variables)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(expected_temp_variables)),
      make_not_null(
          &get<gr::Tags::Shift<DataVector, 3>>(expected_temp_variables)),
      make_not_null(&get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(&get<gr::Tags::DetSpatialMetric<DataVector>>(
          expected_temp_variables)),
      make_not_null(&get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(
          expected_temp_variables)),
      make_not_null(&get<gr::Tags::InverseSpacetimeMetric<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelFirstKind<DataVector, 3>>(
              expected_temp_variables)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelSecondKind<DataVector, 3>>(
              expected_temp_variables)),
      make_not_null(
          &get<gr::Tags::TraceSpacetimeChristoffelFirstKind<DataVector, 3>>(
              expected_temp_variables)),
      make_not_null(&get<gr::Tags::SpacetimeNormalVector<DataVector, 3>>(
          expected_temp_variables)),
      // GH gradient tags
      get<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      // GH argument tags
      tuples::get<gr::Tags::SpacetimeMetric<DataVector, 3>>(arg_variables),
      tuples::get<gh::Tags::Pi<DataVector, 3>>(arg_variables),
      tuples::get<gh::Tags::Phi<DataVector, 3>>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma0>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma1>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma2>(arg_variables),

      *tuples::get<gh::gauges::Tags::GaugeCondition>(arg_variables),

      tuples::get<domain::Tags::Mesh<3>>(arg_variables),
      tuples::get<::Tags::Time>(arg_variables),
      tuples::get<domain::Tags::Coordinates<3, Frame::Inertial>>(arg_variables),
      tuples::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                Frame::Inertial>>(
          arg_variables),
      tuples::get<domain::Tags::MeshVelocity<3, Frame::Inertial>>(
          arg_variables));

  // The time derivative function for CurvedScalarWave is
  CurvedScalarWave::TimeDerivative<3>::apply(
      // Scalar evolved variables
      make_not_null(
          &get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(expected_dt_variables)),
      make_not_null(
          &get<::Tags::dt<CurvedScalarWave::Tags::Pi>>(expected_dt_variables)),
      make_not_null(&get<::Tags::dt<CurvedScalarWave::Tags::Phi<3>>>(
          expected_dt_variables)),
      // Scalar temporaries
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(expected_temp_variables)),
      make_not_null(
          &get<gr::Tags::Shift<DataVector, 3>>(expected_temp_variables)),
      make_not_null(&get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
          expected_temp_variables)),
      make_not_null(&get<CurvedScalarWave::Tags::ConstraintGamma1>(
          expected_temp_variables)),
      make_not_null(&get<CurvedScalarWave::Tags::ConstraintGamma2>(
          expected_temp_variables)),
      // Scalar gradient tags
      get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<CurvedScalarWave::Tags::Pi, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<CurvedScalarWave::Tags::Phi<3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      // Scalar argument tags
      tuples::get<CurvedScalarWave::Tags::Pi>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::Phi<3>>(arg_variables),

      tuples::get<gr::Tags::Lapse<DataVector>>(arg_variables),
      tuples::get<gr::Tags::Shift<DataVector, 3>>(arg_variables),
      tuples::get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                Frame::Inertial>>(arg_variables),
      tuples::get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                                Frame::Inertial>>(arg_variables),
      tuples::get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(arg_variables),
      tuples::get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
          arg_variables),
      tuples::get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::ConstraintGamma1>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::ConstraintGamma2>(arg_variables));

  // We compute the trace-reversed stress energy tensor for the expected
  // variables
  ScalarTensor::trace_reversed_stress_energy(
      make_not_null(
          &get<ScalarTensor::Tags::TraceReversedStressEnergy<
              DataVector, 3, ::Frame::Inertial>>(expected_temp_variables)),
      tuples::get<CurvedScalarWave::Tags::Pi>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::Phi<3>>(arg_variables),
      tuples::get<gr::Tags::Lapse<DataVector>>(arg_variables),
      tuples::get<gr::Tags::Shift<DataVector, 3, ::Frame::Inertial>>(
          arg_variables));

  // When we have backreaction we also need to compute and apply the correction
  // to dt pi for the expected variables

  ScalarTensor::add_stress_energy_term_to_dt_pi(
      make_not_null(
          &get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(expected_dt_variables)),
      get<ScalarTensor::Tags::TraceReversedStressEnergy<DataVector, 3,
                                                        ::Frame::Inertial>>(
          expected_temp_variables),
      tuples::get<gr::Tags::Lapse<DataVector>>(arg_variables));

  ScalarTensor::add_scalar_source_to_dt_pi_scalar(
      make_not_null(
          &get<::Tags::dt<CurvedScalarWave::Tags::Pi>>(expected_dt_variables)),
      tuples::get<ScalarTensor::Tags::ScalarSource>(arg_variables),
      tuples::get<gr::Tags::Lapse<DataVector>>(arg_variables));

  // The time derivative function for the combined system is
  ScalarTensor::TimeDerivative::apply(
      // GH evolved variables
      make_not_null(&get<::Tags::dt<gr::Tags::SpacetimeMetric<DataVector, 3>>>(
          dt_variables)),
      make_not_null(
          &get<::Tags::dt<gh::Tags::Pi<DataVector, 3>>>(dt_variables)),
      make_not_null(
          &get<::Tags::dt<gh::Tags::Phi<DataVector, 3>>>(dt_variables)),
      // Scalar evolved variables
      make_not_null(
          &get<::Tags::dt<CurvedScalarWave::Tags::Psi>>(dt_variables)),
      make_not_null(&get<::Tags::dt<CurvedScalarWave::Tags::Pi>>(dt_variables)),
      make_not_null(
          &get<::Tags::dt<CurvedScalarWave::Tags::Phi<3>>>(dt_variables)),
      // GH temporaries
      make_not_null(
          &get<gh::ConstraintDamping::Tags::ConstraintGamma1>(temp_variables)),
      make_not_null(
          &get<gh::ConstraintDamping::Tags::ConstraintGamma2>(temp_variables)),
      make_not_null(&get<gh::Tags::GaugeH<DataVector, 3>>(temp_variables)),
      make_not_null(
          &get<gh::Tags::SpacetimeDerivGaugeH<DataVector, 3>>(temp_variables)),
      make_not_null(&get<gh::Tags::Gamma1Gamma2>(temp_variables)),
      make_not_null(&get<gh::Tags::HalfPiTwoNormals>(temp_variables)),
      make_not_null(
          &get<gh::Tags::NormalDotOneIndexConstraint>(temp_variables)),
      make_not_null(&get<gh::Tags::Gamma1Plus1>(temp_variables)),
      make_not_null(&get<gh::Tags::PiOneNormal<3>>(temp_variables)),
      make_not_null(
          &get<gh::Tags::GaugeConstraint<DataVector, 3>>(temp_variables)),
      make_not_null(&get<gh::Tags::HalfPhiTwoNormals<3>>(temp_variables)),
      make_not_null(
          &get<gh::Tags::ShiftDotThreeIndexConstraint<3>>(temp_variables)),
      make_not_null(&get<gh::Tags::MeshVelocityDotThreeIndexConstraint<3>>(
          temp_variables)),
      make_not_null(&get<gh::Tags::PhiOneNormal<3>>(temp_variables)),
      make_not_null(&get<gh::Tags::PiSecondIndexUp<3>>(temp_variables)),
      make_not_null(
          &get<gh::Tags::ThreeIndexConstraint<DataVector, 3>>(temp_variables)),
      make_not_null(&get<gh::Tags::PhiFirstIndexUp<3>>(temp_variables)),
      make_not_null(&get<gh::Tags::PhiThirdIndexUp<3>>(temp_variables)),
      make_not_null(
          &get<gh::Tags::SpacetimeChristoffelFirstKindThirdIndexUp<3>>(
              temp_variables)),
      make_not_null(&get<gr::Tags::Lapse<DataVector>>(temp_variables)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(temp_variables)),
      make_not_null(
          &get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temp_variables)),
      make_not_null(
          &get<gr::Tags::DetSpatialMetric<DataVector>>(temp_variables)),
      make_not_null(
          &get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(temp_variables)),
      make_not_null(&get<gr::Tags::InverseSpacetimeMetric<DataVector, 3>>(
          temp_variables)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelFirstKind<DataVector, 3>>(
              temp_variables)),
      make_not_null(
          &get<gr::Tags::SpacetimeChristoffelSecondKind<DataVector, 3>>(
              temp_variables)),
      make_not_null(
          &get<gr::Tags::TraceSpacetimeChristoffelFirstKind<DataVector, 3>>(
              temp_variables)),
      make_not_null(
          &get<gr::Tags::SpacetimeNormalVector<DataVector, 3>>(temp_variables)),
      // Scalar temporaries
      make_not_null(
          &get<CurvedScalarWave::Tags::ConstraintGamma1>(temp_variables)),
      make_not_null(
          &get<CurvedScalarWave::Tags::ConstraintGamma2>(temp_variables)),
      // Extra scalar temporaries
      make_not_null(&get<ScalarTensor::Tags::TraceReversedStressEnergy<
                        DataVector, 3, ::Frame::Inertial>>(temp_variables)),
      // GH gradient tags
      get<::Tags::deriv<gr::Tags::SpacetimeMetric<DataVector, 3>,
                        tmpl::size_t<3>, Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<gh::Tags::Pi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<gh::Tags::Phi<DataVector, 3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      // Scalar gradient tags
      get<::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<CurvedScalarWave::Tags::Pi, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      get<::Tags::deriv<CurvedScalarWave::Tags::Phi<3>, tmpl::size_t<3>,
                        Frame::Inertial>>(gradient_variables),
      // GH argument tags
      tuples::get<gr::Tags::SpacetimeMetric<DataVector, 3>>(arg_variables),

      tuples::get<gh::Tags::Pi<DataVector, 3>>(arg_variables),
      tuples::get<gh::Tags::Phi<DataVector, 3>>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma0>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma1>(arg_variables),
      tuples::get<gh::ConstraintDamping::Tags::ConstraintGamma2>(arg_variables),

      *tuples::get<gh::gauges::Tags::GaugeCondition>(arg_variables),

      tuples::get<domain::Tags::Mesh<3>>(arg_variables),
      tuples::get<::Tags::Time>(arg_variables),
      tuples::get<domain::Tags::Coordinates<3, Frame::Inertial>>(arg_variables),
      tuples::get<domain::Tags::InverseJacobian<3, Frame::ElementLogical,
                                                Frame::Inertial>>(
          arg_variables),
      tuples::get<domain::Tags::MeshVelocity<3, Frame::Inertial>>(
          arg_variables),
      // Scalar argument tags
      tuples::get<CurvedScalarWave::Tags::Pi>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::Phi<3>>(arg_variables),

      tuples::get<gr::Tags::Lapse<DataVector>>(arg_variables),
      tuples::get<gr::Tags::Shift<DataVector, 3>>(arg_variables),
      tuples::get<::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                                Frame::Inertial>>(arg_variables),
      tuples::get<::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                                Frame::Inertial>>(arg_variables),
      tuples::get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(arg_variables),
      tuples::get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
          arg_variables),
      tuples::get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::ConstraintGamma1>(arg_variables),
      tuples::get<CurvedScalarWave::Tags::ConstraintGamma2>(arg_variables),

      // Scalar extra argument tags
      tuples::get<ScalarTensor::Tags::ScalarSource>(arg_variables));

  // Finally we compare
  CHECK_VARIABLES_APPROX(dt_variables, expected_dt_variables);
  CHECK_VARIABLES_APPROX(temp_variables, expected_temp_variables);
}
