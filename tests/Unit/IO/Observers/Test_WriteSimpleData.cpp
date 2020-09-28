// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/Matrix.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/IO/Observers/ObserverHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/Initialize.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "IO/Observer/WriteSimpleData.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TaggedTuple.hpp"

// NOLINTNEXTLINE(google-build-using-namespace)
namespace helpers = TestObservers_detail;

namespace {

struct test_metavariables {
  using component_list =
      tmpl::list<helpers::observer_writer_component<test_metavariables>>;

  using observed_reduction_data_tags = observers::make_reduction_data_tags<
      tmpl::list<helpers::reduction_data_from_doubles>>;

  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Observers.WriteSimpleData", "[Unit][Observers]") {
  using obs_writer = helpers::observer_writer_component<test_metavariables>;

  tuples::TaggedTuple<observers::Tags::ReductionFileName,
                      observers::Tags::VolumeFileName>
      cache_data{};
  tuples::get<observers::Tags::VolumeFileName>(cache_data) =
      "./Unit.IO.Observers.WriteSimpleData";
  const auto& output_file_prefix =
      tuples::get<observers::Tags::VolumeFileName>(cache_data);
  ActionTesting::MockRuntimeSystem<test_metavariables> runner{cache_data};
  ActionTesting::emplace_component<obs_writer>(&runner, 0);
  for(size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<obs_writer>(make_not_null(&runner), 0);
  }
  runner.set_phase(test_metavariables::Phase::Testing);

  const std::string h5_file_name = output_file_prefix + "0.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<double> value_dist{1.0, 5.0};
  const size_t vector_size = 10;
  std::vector<double> data_row(vector_size);
  fill_with_random_values(make_not_null(&data_row), make_not_null(&gen),
                          make_not_null(&value_dist));
  std::vector<std::string> legend(vector_size);
  for (size_t i = 0; i < vector_size; ++i) {
    legend[i] = std::to_string(i);
  }
  ActionTesting::threaded_action<obs_writer,
                                 observers::ThreadedActions::WriteSimpleData>(
      make_not_null(&runner), 0, legend, data_row, "/simple_data.dat");
  // scoped to close the file
  {
    h5::H5File<h5::AccessType::ReadOnly> read_file{h5_file_name};
    const auto& dataset = read_file.get<h5::Dat>("/simple_data");
    const auto cols = alg::iota(std::vector<size_t>(vector_size), 0_st);
    const Matrix data_matrix = dataset.get_data_subset(cols, 0, 1);
    for (size_t i = 0; i < vector_size; ++i) {
      CHECK(approx(data_matrix(0, i)) == data_row[i]);
    }
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  }
}
