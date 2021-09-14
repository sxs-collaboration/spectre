// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>

#include "ControlSystem/UpdateFunctionOfTime.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Domain/FunctionsOfTime/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <typename Metavariables>
struct TestSingleton {
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = size_t;
  using metavariables = Metavariables;
  using get_const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<domain::Tags::FunctionsOfTime>;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename metavariables::Phase,
                                        metavariables::Phase::Initialization,
                                        tmpl::list<>>>;
};

struct TestingMetavariables {
  enum class Phase { Initialization, Testing, Exit };
  using component_list = tmpl::list<TestSingleton<TestingMetavariables>>;
};

SPECTRE_TEST_CASE("Unit.ControlSystem.UpdateFunctionOfTime",
                  "[Unit][ControlSystem]") {
  constexpr size_t deriv_order = 2;
  const double t0 = 0.0;
  const double expr_time = 1.0;
  const double update_time = 1.5;
  const double new_expiration_time = 2.0;
  domain::FunctionsOfTime::register_derived_with_charm();

  // Construct unordered map
  const std::string pp_name{"TestPiecewisePolynomial"};
  const std::string quatfot_name{"TestQuatFunctionOfTime"};
  std::array<DataVector, deriv_order + 1> init_pp_func{
      {DataVector{3, 0.0}, DataVector{3, 0.0}, DataVector{3, 0.0}}};
  std::array<DataVector, 1> init_quat_func{DataVector{4, 1.0}};

  domain::FunctionsOfTime::PiecewisePolynomial<deriv_order> expected_pp{
      t0, init_pp_func, expr_time};
  domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>
      expected_quatfot{t0, init_quat_func, init_pp_func, expr_time};

  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      f_of_t_map{};
  f_of_t_map[pp_name] = std::make_unique<
      domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>>(
      t0, init_pp_func, expr_time);
  f_of_t_map[quatfot_name] = std::make_unique<
      domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>>(
      t0, init_quat_func, init_pp_func, expr_time);

  const DataVector updated_deriv{3, 1.0};

  // Construct mock system and set to Testing phase
  ActionTesting::MockRuntimeSystem<TestingMetavariables> runsys{
      {}, {std::move(f_of_t_map)}};
  ActionTesting::emplace_singleton_component<
      TestSingleton<TestingMetavariables>>(make_not_null(&runsys),
                                           ActionTesting::NodeId{0},
                                           ActionTesting::LocalCoreId{0});
  ActionTesting::set_phase(make_not_null(&runsys),
                           TestingMetavariables::Phase::Testing);

  // Update functions of time in global cache with new deriv
  auto& cache =
      ActionTesting::cache<TestSingleton<TestingMetavariables>>(runsys, 0_st);

  for (auto& name : {pp_name, quatfot_name}) {
    Parallel::mutate<domain::Tags::FunctionsOfTime,
                     control_system::UpdateFunctionOfTime<deriv_order>>(
        cache, name, update_time, updated_deriv, new_expiration_time);
  }

  // Update expected function of time
  expected_pp.update(update_time, updated_deriv, new_expiration_time);
  expected_quatfot.update(update_time, updated_deriv, new_expiration_time);

  // Check that the FunctionsOfTime are what we expected
  const auto& cache_f_of_t_map = get<domain::Tags::FunctionsOfTime>(cache);
  {
    const auto& cache_pp = dynamic_cast<
        const domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>&>(
        *(cache_f_of_t_map.at(pp_name)));
    const auto& cache_quatfot = dynamic_cast<
        const domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>&>(
        *(cache_f_of_t_map.at(quatfot_name)));

    CHECK(cache_pp == expected_pp);
    CHECK(cache_quatfot == expected_quatfot);
  }
  // Update functions of time in global cache with new expiration time
  const double newer_expiration_time = new_expiration_time + 1.0;
  for (auto& name : {pp_name, quatfot_name}) {
    Parallel::mutate<
        domain::Tags::FunctionsOfTime,
        control_system::ResetFunctionOfTimeExpirationTime<deriv_order>>(
        cache, name, newer_expiration_time);
  }

  // Update expected function of time
  expected_pp.reset_expiration_time(newer_expiration_time);
  expected_quatfot.reset_expiration_time(newer_expiration_time);
  {
    const auto& cache_pp = dynamic_cast<
        const domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>&>(
        *(cache_f_of_t_map.at(pp_name)));
    const auto& cache_quatfot = dynamic_cast<
        const domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>&>(
        *(cache_f_of_t_map.at(quatfot_name)));

    // Check that the FunctionsOfTime and expiration times are what we expected
    CHECK(cache_pp == expected_pp);
    CHECK(cache_quatfot == expected_quatfot);
    CHECK(cache_pp.time_bounds()[1] == newer_expiration_time);
    CHECK(cache_quatfot.time_bounds()[1] == newer_expiration_time);
  }
}
}  // namespace
