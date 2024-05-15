// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Evolution/Systems/Cce/Actions/WriteScriBondiQuantities.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Cce.hpp"
#include "IO/H5/File.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/NodeLock.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace helpers = TestObservers_detail;

namespace {
using registration_list = tmpl::list<>;

using metavariables = helpers::Metavariables<registration_list>;
using obs_component = helpers::observer_component<metavariables>;
using obs_writer = helpers::observer_writer_component<metavariables>;
using element_comp =
    helpers::element_component<metavariables, registration_list>;

void test() {
  const std::string output_file_prefix = "WriteScriBondiQuantities";
  const std::string output_file = output_file_prefix + ".h5";
  if (file_system::check_if_file_exists(output_file)) {
    file_system::rm(output_file, true);
  }
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {Domain<3>{},
       std::unordered_map<
           std::string,
           std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>{},
       output_file_prefix, ""}};
  ActionTesting::emplace_nodegroup_component<obs_writer>(&runner);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0);
  auto& obs_writer_proxy = Parallel::get_parallel_component<obs_writer>(cache);

  const size_t l_max = 12;
  const size_t num_points = 2 * square(l_max + 1) + 1;

  // We don't actually care here about what data was written, just that it *was*
  // written
  std::unordered_set<std::string> bondi_variables{"EthInertialRetardedTime",
                                                  "News",
                                                  "Psi0",
                                                  "Psi1",
                                                  "Psi2",
                                                  "Psi3",
                                                  "Psi4",
                                                  "Strain"};
  std::unordered_map<std::string, std::vector<double>> fake_data{};
  for (const std::string& bondi_var : bondi_variables) {
    fake_data[bondi_var] = std::vector<double>(num_points, 0.0);
  }

  // First test the threaded action
  Parallel::threaded_action<Cce::Actions::WriteScriBondiQuantities>(
      obs_writer_proxy[0], "/TestBondiThreaded", l_max, fake_data);

  CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer>(runner,
                                                                     0) == 1);
  ActionTesting::invoke_queued_threaded_action<obs_writer>(
      make_not_null(&runner), 0);

  {
    h5::H5File<h5::AccessType::ReadOnly> h5_file{output_file};
    const auto groups = h5_file.groups();
    REQUIRE(alg::found(groups, "TestBondiThreaded.cce"));
    const auto& cce_file = h5_file.get<h5::Cce>("TestBondiThreaded", l_max, 0);
    const std::unordered_map<std::string, Matrix> data_matrix =
        cce_file.get_data();
    for (const std::string& bondi_var : bondi_variables) {
      REQUIRE(data_matrix.contains(bondi_var));
      CHECK(data_matrix.at(bondi_var).rows() == 1);
      CHECK(data_matrix.at(bondi_var).columns() == num_points);
    }
  }

  // Now test the local synchronous action. However, the action testing
  // framework can't handle local synchronous actions so we just call the apply
  // function itself.
  auto& box = ActionTesting::get_databox<obs_writer>(make_not_null(&runner), 0);
  Parallel::NodeLock lock{};

  Cce::Actions::WriteScriBondiQuantities::apply<void>(
      box, make_not_null(&lock), cache, "/TestBondiSync", l_max, fake_data);

  {
    h5::H5File<h5::AccessType::ReadOnly> h5_file{output_file};
    const auto groups = h5_file.groups();
    REQUIRE(alg::found(groups, "TestBondiSync.cce"));
    const auto& cce_file = h5_file.get<h5::Cce>("TestBondiSync", l_max, 0);
    const std::unordered_map<std::string, Matrix> data_matrix =
        cce_file.get_data();
    for (const std::string& bondi_var : bondi_variables) {
      REQUIRE(data_matrix.contains(bondi_var));
      CHECK(data_matrix.at(bondi_var).rows() == 1);
      CHECK(data_matrix.at(bondi_var).columns() == num_points);
    }
  }

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(Cce::Actions::WriteScriBondiQuantities::apply<void>(
                        box, make_not_null(&lock), cache, "/TestBondiFail",
                        l_max + 1, fake_data),
                    Catch::Matchers::ContainsSubstring(
                        "Some data sent to WriteScriBondiQuantities is "
                        "not of the proper size 393"));
#endif

  if (file_system::check_if_file_exists(output_file)) {
    file_system::rm(output_file, true);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Actions.WriteScriBondiQuantities",
                  "[Unit][Cce]") {
  test();
}
