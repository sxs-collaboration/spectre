// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/optional.hpp>
#include <boost/optional/optional_io.hpp>
#include <cstddef>
#include <memory>
#include <tuple>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhInterfaceManager.hpp"
#include "Evolution/Systems/Cce/InterfaceManagers/GhLockstep.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Literals.hpp"

namespace Cce {
namespace {

template <typename Generator>
void test_gh_lockstep_interface_manager(
    const gsl::not_null<Generator*> gen) noexcept {
  UniformCustomDistribution<double> value_dist{-5.0, 5.0};
  UniformCustomDistribution<size_t> timestep_dist{1, 5};

  std::vector<std::tuple<TimeStepId,
                         InterfaceManagers::GhInterfaceManager::gh_variables>>
      expected_gh_data(7);
  size_t running_total = 0;
  InterfaceManagers::GhLockstep interface_manager{};
  CHECK(interface_manager.get_interpolation_strategy() ==
        InterfaceManagers::InterpolationStrategy::EverySubstep);
  tnsr::aa<DataVector, 3> spacetime_metric{5_st};
  tnsr::iaa<DataVector, 3> phi{5_st};
  tnsr::aa<DataVector, 3> pi{5_st};
  // insert some time ids
  for (size_t i = 0; i < 7; ++i) {
    const size_t substep = running_total % 3;
    const size_t step = running_total / 3;
    // RK3-style substep
    int substep_numerator = 0;
    if (substep == 1) {
      substep_numerator = 2;
    } else if (substep == 2) {
      substep_numerator = 1;
    }
    const Time step_time{{static_cast<double>(step), step + 1.0}, {0, 1}};
    const Time substep_time{{static_cast<double>(step), step + 1.0},
                            {substep_numerator, 2}};
    const TimeStepId time_id{true, static_cast<int64_t>(step), step_time,
                             substep, substep_time};
    fill_with_random_values(make_not_null(&spacetime_metric), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&phi), gen,
                            make_not_null(&value_dist));
    fill_with_random_values(make_not_null(&pi), gen,
                            make_not_null(&value_dist));
    interface_manager.insert_gh_data(time_id, spacetime_metric, phi, pi);
    InterfaceManagers::GhInterfaceManager::gh_variables vars{
        get<0, 0>(spacetime_metric).size()};
    get<gr::Tags::SpacetimeMetric<3, ::Frame::Inertial, DataVector>>(vars) =
        spacetime_metric;
    get<GeneralizedHarmonic::Tags::Pi<3, ::Frame::Inertial>>(vars) = pi;
    get<GeneralizedHarmonic::Tags::Phi<3, ::Frame::Inertial>>(vars) = phi;
    expected_gh_data[i] = std::make_tuple(time_id, std::move(vars));
    running_total += timestep_dist(*gen);
  }

  const auto check_data_retrieval_against_vector =
      [&expected_gh_data](
          const gsl::not_null<InterfaceManagers::GhInterfaceManager*>
              local_interface_manager,
          const size_t expected_number_of_gh_times,
          const size_t vector_index) noexcept {
        CHECK(local_interface_manager->number_of_pending_requests() == 0);
        CHECK(local_interface_manager->number_of_gh_times() ==
              expected_number_of_gh_times);
        auto retrieved_data =
            local_interface_manager->retrieve_and_remove_first_ready_gh_data();
        REQUIRE(retrieved_data);
        CHECK(get<0>(*retrieved_data) ==
              get<0>(expected_gh_data[vector_index]));
        CHECK(get<1>(*retrieved_data) ==
              get<1>(expected_gh_data[vector_index]));

        CHECK(local_interface_manager->number_of_pending_requests() == 0);
        CHECK(local_interface_manager->number_of_gh_times() ==
              expected_number_of_gh_times - 1);
      };
  {
    INFO("Retrieve data from directly constructed manager");
    // choose a timestep to request - requests should do nothing
    interface_manager.request_gh_data(get<0>(expected_gh_data[1]));
    interface_manager.request_gh_data(get<0>(expected_gh_data[2]));
    check_data_retrieval_against_vector(make_not_null(&interface_manager), 7_st,
                                        0_st);
    // add some of the individual next_times -- this should also do nothing for
    // the lockstep interface manager
    interface_manager.insert_next_gh_time(get<0>(expected_gh_data[1]),
                                          get<0>(expected_gh_data[2]));
    interface_manager.insert_next_gh_time(get<0>(expected_gh_data[2]),
                                          get<0>(expected_gh_data[3]));
    check_data_retrieval_against_vector(make_not_null(&interface_manager), 6_st,
                                        1_st);
  }
  {
    INFO("Retrieve data from serialized and deserialized manager");
    // check that the state is preserved during serialization
    auto serialized_and_deserialized_interface_manager =
        serialize_and_deserialize(interface_manager);
    serialized_and_deserialized_interface_manager.request_gh_data(
        get<0>(expected_gh_data[2]));
    check_data_retrieval_against_vector(
        make_not_null(&serialized_and_deserialized_interface_manager), 5_st,
        2_st);
  }

  {
    INFO("Retrieve data from cloned unique_ptr to manager");
    // check that the state is preserved through cloning
    auto cloned_interface_manager = interface_manager.get_clone();
    cloned_interface_manager->request_gh_data(get<0>(expected_gh_data[2]));
    check_data_retrieval_against_vector(
        make_not_null(cloned_interface_manager.get()), 5_st, 2_st);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.GhLockstep",
                  "[Unit][Cce]") {
  MAKE_GENERATOR(gen);
  test_gh_lockstep_interface_manager(make_not_null(&gen));
}
}  // namespace Cce
