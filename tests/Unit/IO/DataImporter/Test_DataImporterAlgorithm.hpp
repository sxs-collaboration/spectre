// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#define CATCH_CONFIG_RUNNER

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "AlgorithmArray.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "IO/DataImporter/DataFileReader.hpp"
#include "IO/DataImporter/DataFileReaderActions.hpp"
#include "IO/DataImporter/ElementActions.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarField"; }
};

template <size_t Dim>
struct VectorFieldTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() noexcept { return "VectorField"; }
};

struct NumericTestData {
  using group = importer::OptionTags::Group;
  static constexpr OptionString help = "Numeric data";
};

static constexpr size_t number_of_elements = 2;
static constexpr std::array<std::array<size_t, 3>, number_of_elements> extents{
    {// Grid extents on first element
     {{2, 1, 1}},
     // Grid extents on second element
     {{3, 1, 1}}}};
static const std::array<DataVector, number_of_elements> scalar_field_data{
    {// Field on first element
     {{1., 2.}},
     // Field on second element
     {{3., 4., 5.}}}};
static const std::array<std::array<DataVector, 3>, number_of_elements>
    vector_field_data{
        {// Vector components on first element
         {{{{1., 2.}}, {{3., 4.}}, {{5., 6.}}}},
         // Vector components on second element
         {{{{7., 8., 9.}}, {{10., 11., 12.}}, {{13., 14., 16.}}}}}};

template <size_t Dim>
struct WriteTestData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    // Open file for test data
    const auto& data_file_name =
        get<importer::Tags::DataFileName<NumericTestData>>(cache);
    if (file_system::check_if_file_exists(data_file_name)) {
      file_system::rm(data_file_name, true);
    }
    h5::H5File<h5::AccessType::ReadWrite> data_file{data_file_name};
    auto& test_data_file = data_file.insert<h5::VolumeData>("/test_data");

    // Construct test data for all elements
    std::vector<ExtentsAndTensorVolumeData> element_data{};
    for (size_t i = 0; i < number_of_elements; i++) {
      const std::string element_name = MakeString{} << ElementId<Dim>{i};
      std::vector<TensorComponent> tensor_components{};
      std::vector<size_t> element_extents{};
      tensor_components.push_back(
          {element_name + "/ScalarField", scalar_field_data[i]});
      for (size_t d = 0; d < Dim; d++) {
        static const std::array<std::string, 3> dim_suffix{{"x", "y", "z"}};
        tensor_components.push_back(
            {element_name + "/VectorField_" + dim_suffix[d],
             vector_field_data[i][d]});
        element_extents.push_back(extents[i][d]);
      }
      element_data.push_back(
          {std::move(element_extents), std::move(tensor_components)});
    }
    test_data_file.write_volume_data(0, 3., std::move(element_data));

    return {std::move(box), true};
  }
};

struct CleanTestData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const auto& data_file_name =
        get<importer::Tags::DataFileName<NumericTestData>>(cache);
    if (file_system::check_if_file_exists(data_file_name)) {
      // file_system::rm(data_file_name, true);
    } else {
      ERROR("Expected test data file '" << data_file_name
                                        << "' does not exist");
    }
    return {std::move(box), true};
  }
};

template <size_t Dim, typename Metavariables>
struct TestDataWriter {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::Initialization,
                                        tmpl::list<WriteTestData<Dim>>>,

                 Parallel::PhaseActions<typename Metavariables::Phase,
                                        Metavariables::Phase::TestResult,
                                        tmpl::list<CleanTestData>>>;
  using initialization_tags = tmpl::list<>;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(Parallel::CProxy_ConstGlobalCache<
                         Metavariables>& /*global_cache*/) noexcept {}

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_component = Parallel::get_parallel_component<TestDataWriter>(
        *(global_cache.ckLocalBranch()));
    local_component.start_phase(next_phase);
  }
};

template <size_t Dim>
struct InitializeElement {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    return std::make_tuple(
        ::Initialization::merge_into_databox<
            InitializeElement,
            db::AddSimpleTags<ScalarFieldTag, VectorFieldTag<Dim>>>(
            std::move(box), Scalar<DataVector>{}, tnsr::I<DataVector, Dim>{}),
        true);
  }
};

template <size_t Dim>
struct TestResult {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static std::tuple<db::DataBox<DbTagsList>&&, bool> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ElementIndex<Dim>& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    const size_t raw_element_index = ElementId<Dim>{array_index}.block_id();
    Scalar<DataVector> expected_scalar_field{
        scalar_field_data[raw_element_index]};
    SPECTRE_PARALLEL_REQUIRE(get<ScalarFieldTag>(box) == expected_scalar_field);
    tnsr::I<DataVector, Dim> expected_vector_field{};
    for (size_t d = 0; d < Dim; d++) {
      expected_vector_field[d] = vector_field_data[raw_element_index][d];
    }
    SPECTRE_PARALLEL_REQUIRE(get<VectorFieldTag<Dim>>(box) ==
                             expected_vector_field);
    return {std::move(box), true};
  }
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using array_index = ElementIndex<Dim>;
  using metavariables = Metavariables;
  using initialization_tags = tmpl::list<>;
  using array_allocation_tags = tmpl::list<>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Initialization,
                             tmpl::list<InitializeElement<Dim>>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Register,
                             tmpl::list<importer::Actions::RegisterWithImporter,
                                        Parallel::Actions::TerminatePhase>>,

      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::TestResult,
                             tmpl::list<TestResult<Dim>>>>;

  using import_fields = tmpl::list<ScalarFieldTag, VectorFieldTag<Dim>>;
  using read_element_data_action = importer::ThreadedActions::ReadElementData<
      NumericTestData, import_fields, ::Actions::SetData<import_fields>,
      ElementArray>;

  using const_global_cache_tag_list =
      typename read_element_data_action::const_global_cache_tag_list;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
          initialization_items) noexcept {
    auto& array_proxy = Parallel::get_parallel_component<ElementArray>(
        *(global_cache.ckLocalBranch()));

    for (size_t i = 0, which_proc = 0,
                number_of_procs =
                    static_cast<size_t>(Parallel::number_of_procs());
         i < number_of_elements; i++) {
      ElementIndex<Dim> element_index{ElementId<Dim>{i}};
      array_proxy[element_index].insert(global_cache, initialization_items,
                                        which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<ElementArray>(local_cache)
        .start_phase(next_phase);

    if (next_phase == Metavariables::Phase::ImportData) {
      Parallel::threaded_action<read_element_data_action>(
          Parallel::get_parallel_component<
              importer::DataFileReader<Metavariables>>(local_cache));
    }
  }
};

}  // namespace

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>,
                                    TestDataWriter<Dim, Metavariables>,
                                    importer::DataFileReader<Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;

  static constexpr const char* const help{"Test the data importer"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  enum class Phase { Initialization, Register, ImportData, TestResult, Exit };

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::Register;
      case Phase::Register:
        return Phase::ImportData;
      case Phase::ImportData:
        return Phase::TestResult;
      default:
        return Phase::Exit;
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
