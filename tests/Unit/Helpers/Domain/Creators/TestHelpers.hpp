// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Tuple.hpp"

namespace TestHelpers::domain::creators {

template <size_t Dim>
Domain<Dim> test_domain_creator(const DomainCreator<Dim>& domain_creator,
                                const bool expect_boundary_conditions,
                                const bool is_periodic = false,
                                const std::vector<double>& times = {
                                    // quiet NaN so CAPTURE(time) works
                                    std::numeric_limits<double>::quiet_NaN()}) {
  INFO("Test domain creator consistency");
  CAPTURE(Dim);
  auto domain = domain_creator.create_domain();
  const auto block_names = domain_creator.block_names();
  const auto block_groups = domain_creator.block_groups();
  const auto all_boundary_conditions =
      domain_creator.external_boundary_conditions();
  const auto initial_refinement_levels =
      domain_creator.initial_refinement_levels();
  const auto initial_extents = domain_creator.initial_extents();

  const auto& blocks = domain.blocks();
  REQUIRE(initial_refinement_levels.size() == blocks.size());
  REQUIRE(initial_extents.size() == blocks.size());
  {
    CAPTURE(block_names);
    CHECK((block_names.empty() or (block_names.size() == blocks.size())));
    for (size_t block_id = 0; block_id < block_names.size(); ++block_id) {
      CHECK(blocks[block_id].name() == block_names[block_id]);
    }
    CHECK(domain.block_groups() == block_groups);
    {
      INFO("Test block names are unique");
      auto sorted_block_names = block_names;
      alg::sort(sorted_block_names);
      CHECK(std::adjacent_find(sorted_block_names.begin(),
                               sorted_block_names.end()) ==
            sorted_block_names.end());
    }
    {
      INFO("Test block groups contain valid block names");
      for (const auto& [block_group, block_names_in_group] : block_groups) {
        CAPTURE(block_group);
        for (const auto& block_name : block_names_in_group) {
          CAPTURE(block_name);
          CHECK(alg::find(block_names, block_name) != block_names.end());
        }
      }
    }
    {
      INFO(
          "Test block neighbors are never in the same direction as external "
          "boundaries")
      for (size_t block_id = 0; block_id < block_names.size(); ++block_id) {
        for (const auto& neighbor : blocks[block_id].neighbors()) {
          // external and neighbor directions should never match
          const auto& external_boundaries =
              blocks[block_id].external_boundaries();
          CHECK(external_boundaries.find(neighbor.first) ==
                external_boundaries.end());
        }
      }
    }
  }

  ::domain::creators::register_derived_with_charm();
  ::domain::creators::time_dependence::register_derived_with_charm();
  test_serialization(domain);

  test_initial_domain(domain, initial_refinement_levels);
  const auto functions_of_time = domain_creator.functions_of_time();
  for (const double time : times) {
    CAPTURE(time);
    if (not is_periodic) {
      test_physical_separation(domain.blocks(), time, functions_of_time);
    }
    // The 1D RotatedIntervals domain creator violates this condition
    if constexpr (Dim != 1) {
      test_det_jac_positive(domain.blocks(), time, functions_of_time);
    }
  }

  if (expect_boundary_conditions) {
    INFO("Boundary conditions");
    REQUIRE(all_boundary_conditions.size() == blocks.size());
    for (size_t block_id = 0; block_id < blocks.size(); ++block_id) {
      CAPTURE(block_id);
      const auto& block = blocks[block_id];
      const auto& boundary_conditions = all_boundary_conditions[block_id];
      const auto& external_boundaries = block.external_boundaries();
      REQUIRE(boundary_conditions.size() == external_boundaries.size());
      for (const auto& direction : Direction<Dim>::all_directions()) {
        CAPTURE(direction);
        if (external_boundaries.find(direction) == external_boundaries.end()) {
          INFO("Internal boundary should not specify a boundary condition");
          CHECK(boundary_conditions.find(direction) ==
                boundary_conditions.end());
        } else {
          INFO("External boundary is missing a boundary condition");
          REQUIRE(boundary_conditions.find(direction) !=
                  boundary_conditions.end());
          REQUIRE(boundary_conditions.at(direction) != nullptr);
        }
      }
    }
  } else {
    CHECK(all_boundary_conditions.empty());
  }

  // Check that every direction in every excision_sphere is also an
  // external boundary of the correct Block.
  for (const auto& excision_sphere_map_element : domain.excision_spheres()) {
    for (const auto& [block_index, direction] :
         excision_sphere_map_element.second.abutting_directions()) {
      const auto& external_boundaries =
          domain.blocks()[block_index].external_boundaries();
      CHECK(external_boundaries.find(direction) != external_boundaries.end());
    }
  }

  return domain;
}

template <size_t Dim, typename... ExpectedFunctionsOfTime>
void test_functions_of_time(
    const DomainCreator<Dim>& creator,
    const std::tuple<std::pair<std::string, ExpectedFunctionsOfTime>...>&
        expected_functions_of_time,
    const std::unordered_map<std::string, double>& initial_expiration_times =
        {}) {
  const std::unordered_map<
      std::string, std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
      functions_of_time = creator.functions_of_time(initial_expiration_times);
  REQUIRE(functions_of_time.size() == sizeof...(ExpectedFunctionsOfTime));

  tuple_fold(expected_functions_of_time,
             [&functions_of_time](const auto& name_and_function_of_time) {
               const std::string& name = name_and_function_of_time.first;
               const auto& function_of_time = name_and_function_of_time.second;
               using FunctionOfTimeType =
                   std::decay_t<decltype(function_of_time)>;
               const bool in_functions_of_time =
                   functions_of_time.find(name) != functions_of_time.end();
               // NOLINTNEXTLINE(bugprone-infinite-loop) false positive
               CHECK(in_functions_of_time);
               if (in_functions_of_time) {
                 const auto* function_from_creator =
                     dynamic_cast<const FunctionOfTimeType*>(
                         functions_of_time.at(name).get());
                 REQUIRE(function_from_creator != nullptr);
                 CHECK(*function_from_creator == function_of_time);
               }
             });
}
}  // namespace TestHelpers::domain::creators
