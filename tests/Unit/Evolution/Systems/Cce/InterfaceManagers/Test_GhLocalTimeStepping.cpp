// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLocalTimeStepping.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Interpolation/BarycentricRationalSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/CubicSpanInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/LinearSpanInterpolator.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Literals.hpp"

namespace Cce::InterfaceManagers {
namespace {

template <typename Generator>
void test_gh_local_time_stepping_interface_manager(
    const gsl::not_null<Generator*> gen) noexcept {
  Parallel::register_derived_classes_with_charm<intrp::SpanInterpolator>();
  // the frequency has to be small to be kind to the time stepper for the ~.1
  // step size in this test
  UniformCustomDistribution<double> value_dist{0.1, 1.0};
  const double frequency = value_dist(*gen);

  const Slab source_slab{0.0, 0.01};
  // target slab twice as large to verify that the details of the TimeStepId's
  // don't matter to the interface manager
  const Slab target_slab{0.0, 0.02};

  const std::vector<TimeStepId> source_time_steps{
      {true, 0, {source_slab, {0, 1}}}, {true, 0, {source_slab, {1, 10}}},
      {true, 0, {source_slab, {1, 5}}}, {true, 0, {source_slab, {3, 10}}},
      {true, 0, {source_slab, {2, 5}}}, {true, 0, {source_slab, {1, 2}}},
      {true, 0, {source_slab, {3, 5}}}, {true, 0, {source_slab, {7, 10}}},
      {true, 0, {source_slab, {4, 5}}}, {true, 0, {source_slab, {9, 10}}}};
  const std::vector<TimeStepId> target_time_steps{
      {true, 0, {target_slab, {0, 1}}},   {true, 0, {target_slab, {1, 40}}},
      {true, 0, {target_slab, {3, 40}}},  {true, 0, {target_slab, {7, 40}}},
      {true, 0, {target_slab, {19, 80}}}, {true, 0, {target_slab, {7, 20}}},
      {true, 0, {target_slab, {15, 40}}}, {true, 0, {target_slab, {1, 2}}}};
  InterfaceManagers::GhLocalTimeStepping interface_manager{
      std::make_unique<intrp::BarycentricRationalSpanInterpolator>(2u, 2u)};

  // These represent data at time = 0, the time dependence for item i in the
  // vector will be a * cos((frequency + i * 0.05) * t), so the first derivative
  // is -a * (frequency + i * 0.05) * sin((frequency + i * 0.05) * t)
  const double frequency_increment = 5.0e-2;
  tnsr::aa<DataVector, 3> spacetime_metric{5_st};
  tnsr::iaa<DataVector, 3> phi{5_st};
  tnsr::aa<DataVector, 3> pi{5_st};
  fill_with_random_values(make_not_null(&spacetime_metric), gen,
                          make_not_null(&value_dist));
  fill_with_random_values(make_not_null(&phi), gen, make_not_null(&value_dist));
  fill_with_random_values(make_not_null(&pi), gen, make_not_null(&value_dist));

  const auto check_no_retrieval =
      [](const gsl::not_null<InterfaceManagers::GhLocalTimeStepping*>
             local_interface_manager) noexcept {
        CHECK_FALSE(
            static_cast<bool>(local_interface_manager
                                  ->retrieve_and_remove_first_ready_gh_data()));
      };

  const auto insert_source_data =
      [&source_time_steps, &spacetime_metric, &phi, &pi, &frequency,
       &frequency_increment](
          const gsl::not_null<InterfaceManagers::GhLocalTimeStepping*>
              local_interface_manager,
          const size_t index) noexcept {
        const auto current_time_step_id = source_time_steps[index];
        const double current_time = current_time_step_id.substep_time().value();

        tnsr::aa<DataVector, 3> current_spacetime_metric{5_st};
        tnsr::iaa<DataVector, 3> current_phi{5_st};
        tnsr::aa<DataVector, 3> current_pi{5_st};

        tnsr::aa<DataVector, 3> current_dt_spacetime_metric{5_st};
        tnsr::iaa<DataVector, 3> current_dt_phi{5_st};
        tnsr::aa<DataVector, 3> current_dt_pi{5_st};
        for (size_t i = 0; i < tnsr::aa<DataVector, 3>::size(); ++i) {
          for (size_t j = 0; j < current_spacetime_metric[i].size(); ++j) {
            current_spacetime_metric[i][j] =
                spacetime_metric[i][j] *
                cos((frequency + j * frequency_increment) * current_time);

            current_pi[i][j] =
                pi[i][j] *
                cos((frequency + j * frequency_increment) * current_time);
          }
        }
        for (size_t i = 0; i < tnsr::iaa<DataVector, 3>::size(); ++i) {
          for (size_t j = 0; j < current_phi[i].size(); ++j) {
            current_phi[i][j] =
                phi[i][j] *
                cos((frequency + j * frequency_increment) * current_time);
          }
        }
        local_interface_manager->insert_gh_data(
            current_time_step_id.substep_time().value(),
            current_spacetime_metric, current_phi, current_pi);
      };

  const auto request_target_time =
      [&target_time_steps](
          const gsl::not_null<InterfaceManagers::GhLocalTimeStepping*>
              local_interface_manager,
          const size_t index) noexcept {
        local_interface_manager->request_gh_data(target_time_steps[index]);
      };

  const auto check_retrieval = [&target_time_steps, &frequency,
                                &frequency_increment, &spacetime_metric, &pi,
                                &phi](
                                   const gsl::not_null<
                                       InterfaceManagers::GhLocalTimeStepping*>
                                       local_interface_manager,
                                   const size_t index,
                                   Approx local_approx = approx) noexcept {
    const double current_time = target_time_steps[index].substep_time().value();
    tnsr::aa<DataVector, 3> expected_spacetime_metric{5_st};
    tnsr::iaa<DataVector, 3> expected_phi{5_st};
    tnsr::aa<DataVector, 3> expected_pi{5_st};

    for (size_t i = 0; i < tnsr::aa<DataVector, 3>::size(); ++i) {
      for (size_t j = 0; j < expected_spacetime_metric[i].size(); ++j) {
        expected_spacetime_metric[i][j] =
            spacetime_metric[i][j] *
            cos((frequency + j * frequency_increment) * current_time);

        expected_pi[i][j] =
            pi[i][j] *
            cos((frequency + j * frequency_increment) * current_time);
      }
    }
    for (size_t i = 0; i < tnsr::iaa<DataVector, 3>::size(); ++i) {
      for (size_t j = 0; j < expected_phi[i].size(); ++j) {
        expected_phi[i][j] =
            phi[i][j] *
            cos((frequency + j * frequency_increment) * current_time);
      }
    }

    const auto retrieved_data =
        local_interface_manager->retrieve_and_remove_first_ready_gh_data();
    REQUIRE(static_cast<bool>(retrieved_data));
    CHECK_ITERABLE_CUSTOM_APPROX(
        SINGLE_ARG(
            get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(
                get<1>(*retrieved_data))),
        expected_spacetime_metric, local_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        SINGLE_ARG(get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(
            get<1>(*retrieved_data))),
        expected_pi, local_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        SINGLE_ARG(get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(
            get<1>(*retrieved_data))),
        expected_phi, local_approx);
  };

  // Test plan (given in ratios of the source interval):
  // insert 0.0
  // request 0.0
  // fail to retrieve 0.0
  // request 0.05
  // insert 0.1
  // request 0.15
  //
  // insert 0.2
  // request 0.35
  // fail to retrieve 0.0
  // insert 0.3
  // retrieve 0.0
  // retrieve 0.05
  // retrieve 0.15
  // fail to retrieve 0.35
  // insert 0.4
  //
  // request 0.475 (19/40)
  // fail to retrieve .475
  // insert .5 data
  // retrieve .475
  // fail to retrieve no requests
  // insert .6 data
  // request .7
  //
  // clone and serialize; check remaining for original, serialized, and clone
  //
  // request .75
  // insert .7 data
  // insert .8 data
  // retrieve .7
  // retrieve .75

  check_no_retrieval(make_not_null(&interface_manager));

  insert_source_data(make_not_null(&interface_manager), 0_st);
  request_target_time(make_not_null(&interface_manager), 0_st);
  check_no_retrieval(make_not_null(&interface_manager));
  request_target_time(make_not_null(&interface_manager), 1_st);
  insert_source_data(make_not_null(&interface_manager), 1_st);
  request_target_time(make_not_null(&interface_manager), 2_st);

  insert_source_data(make_not_null(&interface_manager), 2_st);
  request_target_time(make_not_null(&interface_manager), 3_st);
  check_no_retrieval(make_not_null(&interface_manager));
  insert_source_data(make_not_null(&interface_manager), 3_st);
  check_retrieval(make_not_null(&interface_manager), 0_st,
                  Approx::custom().epsilon(1.e-5).scale(1.));
  check_retrieval(make_not_null(&interface_manager), 1_st,
                  Approx::custom().epsilon(1.e-7).scale(1.));
  check_retrieval(make_not_null(&interface_manager), 2_st,
                  Approx::custom().epsilon(1.e-9).scale(1.));
  check_no_retrieval(make_not_null(&interface_manager));
  insert_source_data(make_not_null(&interface_manager), 4_st);
  check_retrieval(make_not_null(&interface_manager), 3_st,
                  Approx::custom().epsilon(1.e-11).scale(1.));

  request_target_time(make_not_null(&interface_manager), 4_st);
  check_no_retrieval(make_not_null(&interface_manager));
  insert_source_data(make_not_null(&interface_manager), 5_st);
  check_retrieval(make_not_null(&interface_manager), 4_st,
                  Approx::custom().epsilon(1.e-11).scale(1.));
  check_no_retrieval(make_not_null(&interface_manager));
  insert_source_data(make_not_null(&interface_manager), 6_st);
  request_target_time(make_not_null(&interface_manager), 5_st);

  const auto second_half_checks =
      [&request_target_time, &check_no_retrieval, &check_retrieval,
       &insert_source_data](
          const gsl::not_null<InterfaceManagers::GhLocalTimeStepping*>
              local_interface_manager) noexcept {
        request_target_time(local_interface_manager, 6_st);
        insert_source_data(local_interface_manager, 7_st);
        insert_source_data(local_interface_manager, 8_st);
        check_retrieval(local_interface_manager, 5_st,
                        Approx::custom().epsilon(1.e-11).scale(1.));
        check_retrieval(local_interface_manager, 6_st,
                        Approx::custom().epsilon(1.e-11).scale(1.));
        check_no_retrieval(local_interface_manager);
      };
  auto clone = interface_manager.get_clone();
  auto serialized_and_deserialized_interface_manager =
      serialize_and_deserialize(interface_manager);
  {
    INFO("Checking original");
    second_half_checks(make_not_null(&interface_manager));
  }
  {
    INFO("Checking clone");
    second_half_checks(make_not_null(
        dynamic_cast<InterfaceManagers::GhLocalTimeStepping*>(&(*clone))));
  }
  {
    INFO("Checking serialized and deserialized");
    second_half_checks(
        make_not_null(&serialized_and_deserialized_interface_manager));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.GhLocalTimeStepping",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_gh_local_time_stepping_interface_manager(make_not_null(&gen));
}
}  // namespace Cce::InterfaceManagers
