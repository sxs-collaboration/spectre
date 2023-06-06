// Distributed under the MIT License.
// See LICENSE.txt for details.

/*!
 * \file
 * \brief Test reading in and interpolating volume data from H5 files
 *
 * Input files specify a source domain and a target domain. Test data on the
 * source domain is constructed and written to H5 files, then read back in and
 * interpolated to the target domain.
 */

#pragma once

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Factory1D.hpp"
#include "Domain/Creators/Factory2D.hpp"
#include "Domain/Creators/Factory3D.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Options/String.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/SetData.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MemoryHelpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "ScalarField"; }
};

template <size_t Dim>
struct VectorFieldTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() { return "VectorField"; }
};

/// The source or target of the interpolation
enum class SourceOrTarget {
  /// The domain to interpolate FROM
  Source,
  /// The domain to interpolate TO
  Target
};

inline std::ostream& operator<<(std::ostream& os,
                                SourceOrTarget source_or_target) {
  switch (source_or_target) {
    case SourceOrTarget::Source:
      return os << "Source";
    case SourceOrTarget::Target:
      return os << "Target";
    default:
      ERROR("Missing case for SourceOrTarget");
  }
}

// Tags for both the source and target domains of the interpolation

namespace OptionTags {
template <size_t Dim, SourceOrTarget WhichDomain>
struct DomainCreator {
  static std::string name() {
    return MakeString{} << WhichDomain << "DomainCreator";
  }
  using type = std::unique_ptr<::DomainCreator<Dim>>;
  static constexpr Options::String help = {"Choose a domain."};
};
}  // namespace OptionTags

namespace Tags {
template <size_t Dim, SourceOrTarget WhichDomain>
struct Domain : db::SimpleTag {
  using type = ::Domain<Dim>;
  using option_tags = tmpl::list<OptionTags::DomainCreator<Dim, WhichDomain>>;
  static constexpr bool pass_metavariables = false;
  static ::Domain<Dim> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->create_domain();
  }
};

template <size_t Dim, SourceOrTarget WhichDomain>
struct FunctionsOfTime : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>;
  using option_tags = tmpl::list<OptionTags::DomainCreator<Dim, WhichDomain>>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->functions_of_time();
  }
};

template <size_t Dim, SourceOrTarget WhichDomain>
struct InitialExtents : db::SimpleTag {
  using type = std::vector<std::array<size_t, Dim>>;
  using option_tags = tmpl::list<OptionTags::DomainCreator<Dim, WhichDomain>>;
  static constexpr bool pass_metavariables = false;
  static std::vector<std::array<size_t, Dim>> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->initial_extents();
  }
};

template <size_t Dim, SourceOrTarget WhichDomain>
struct InitialRefinementLevels : db::SimpleTag {
  using type = std::vector<std::array<size_t, Dim>>;
  using option_tags = tmpl::list<OptionTags::DomainCreator<Dim, WhichDomain>>;
  static constexpr bool pass_metavariables = false;
  static std::vector<std::array<size_t, Dim>> create_from_options(
      const std::unique_ptr<::DomainCreator<Dim>>& domain_creator) {
    return domain_creator->initial_refinement_levels();
  }
};
}  // namespace Tags

/// [option_group]
struct OptionsGroup {
  static std::string name() { return "Importers"; }
  static constexpr Options::String help = "Numeric volume data";
};
/// [option_group]

template <bool Check>
void clean_test_data(const std::string& data_file_name) {
  if (file_system::check_if_file_exists(data_file_name)) {
    file_system::rm(data_file_name, true);
  } else if (Check) {
    ERROR("Expected test data file '" << data_file_name << "' does not exist");
  }
}

template <size_t Dim>
DataVector gaussian(const tnsr::I<DataVector, Dim>& x) {
  return exp(-get(dot_product(x, x)));
}

template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coordinates(
    const ElementId<Dim>& element_id, const Mesh<Dim>& mesh,
    const Block<Dim>& block, const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  const auto logical_coords = logical_coordinates(mesh);
  if (block.is_time_dependent()) {
    const ElementMap<Dim, Frame::Grid> element_map{
        element_id, block.moving_mesh_logical_to_grid_map().get_clone()};
    const auto grid_coords = element_map(logical_coords);
    const auto& grid_to_inertial_map = block.moving_mesh_grid_to_inertial_map();
    return grid_to_inertial_map(grid_coords, time, functions_of_time);
  } else {
    const ElementMap<Dim, Frame::Inertial> element_map{
        element_id, block.stationary_map().get_clone()};
    return element_map(logical_coords);
  }
}

template <size_t Dim>
void write_test_data(
    const std::string& data_file_name, const std::string& subgroup,
    const double observation_value, const ::Domain<Dim>& domain,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const std::vector<std::array<size_t, Dim>>& initial_refinement_levels,
    const std::vector<std::array<size_t, Dim>>& initial_extents) {
  // Open file for test data
  h5::H5File<h5::AccessType::ReadWrite> data_file{data_file_name, true};
  auto& test_data_file = data_file.insert<h5::VolumeData>("/" + subgroup);

  // Construct test data for all elements
  const auto element_ids = initial_element_ids(initial_refinement_levels);
  std::vector<ElementVolumeData> element_data{};
  for (const auto& element_id : element_ids) {
    const auto mesh = domain::Initialization::create_initial_mesh(
        initial_extents, element_id, Spectral::Quadrature::GaussLobatto);
    const size_t num_points = mesh.number_of_grid_points();
    const auto& block = domain.blocks()[element_id.block_id()];
    const auto inertial_coords = inertial_coordinates(
        element_id, mesh, block, observation_value, functions_of_time);

    std::vector<TensorComponent> tensor_components{};
    for (size_t d = 0; d < Dim; ++d) {
      tensor_components.push_back(
          {"InertialCoordinates" + inertial_coords.component_suffix(
                                       inertial_coords.get_tensor_index(d)),
           inertial_coords[d]});
    }
    tensor_components.push_back({"ScalarField", gaussian(inertial_coords)});
    for (size_t d = 0; d < Dim; d++) {
      static const std::array<std::string, 3> dim_suffix{{"x", "y", "z"}};
      tensor_components.push_back(
          {"VectorField_" + dim_suffix[d], DataVector(num_points, 0.)});
    }
    element_data.push_back({element_id, std::move(tensor_components), mesh});
  }
  test_data_file.write_volume_data(0, observation_value,
                                   std::move(element_data), serialize(domain),
                                   serialize(functions_of_time));
}

template <size_t Dim>
struct WriteTestData {
  using const_global_cache_tags =
      tmpl::list<Tags::Domain<Dim, SourceOrTarget::Source>,
                 Tags::FunctionsOfTime<Dim, SourceOrTarget::Source>,
                 Tags::InitialRefinementLevels<Dim, SourceOrTarget::Source>,
                 Tags::InitialExtents<Dim, SourceOrTarget::Source>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& options =
        db::get<importers::Tags::ImporterOptions<OptionsGroup>>(box);
    clean_test_data<false>(get<importers::OptionTags::FileGlob>(options));
    write_test_data<Dim>(
        get<importers::OptionTags::FileGlob>(options),
        get<importers::OptionTags::Subgroup>(options),
        std::get<double>(get<importers::OptionTags::ObservationValue>(options)),
        db::get<Tags::Domain<Dim, SourceOrTarget::Source>>(box),
        db::get<Tags::FunctionsOfTime<Dim, SourceOrTarget::Source>>(box),
        db::get<Tags::InitialRefinementLevels<Dim, SourceOrTarget::Source>>(
            box),
        db::get<Tags::InitialExtents<Dim, SourceOrTarget::Source>>(box));
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct CleanTestData {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& options =
        db::get<importers::Tags::ImporterOptions<OptionsGroup>>(box);
    clean_test_data<true>(get<importers::OptionTags::FileGlob>(options));
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <size_t Dim, typename Metavariables>
struct TestDataWriter {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<Parallel::Phase::Initialization,
                                        tmpl::list<WriteTestData<Dim>>>,

                 Parallel::PhaseActions<Parallel::Phase::Testing,
                                        tmpl::list<CleanTestData>>>;
  using simple_tags_from_options = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_component = Parallel::get_parallel_component<TestDataWriter>(
        *Parallel::local_branch(global_cache));
    local_component.start_phase(next_phase);
  }
};

template <size_t Dim>
struct InitializeElement {
  using const_global_cache_tags =
      tmpl::list<Tags::Domain<Dim, SourceOrTarget::Target>,
                 Tags::FunctionsOfTime<Dim, SourceOrTarget::Target>,
                 Tags::InitialRefinementLevels<Dim, SourceOrTarget::Target>,
                 Tags::InitialExtents<Dim, SourceOrTarget::Target>>;
  using simple_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                 ScalarFieldTag, VectorFieldTag<Dim>>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& domain =
        db::get<Tags::Domain<Dim, SourceOrTarget::Target>>(box);
    const auto& functions_of_time =
        db::get<Tags::FunctionsOfTime<Dim, SourceOrTarget::Target>>(box);
    const auto& options =
        db::get<importers::Tags::ImporterOptions<OptionsGroup>>(box);
    const double time =
        std::get<double>(get<importers::OptionTags::ObservationValue>(options));
    const auto& initial_extents =
        db::get<Tags::InitialExtents<Dim, SourceOrTarget::Target>>(box);
    const auto mesh = domain::Initialization::create_initial_mesh(
        initial_extents, element_id, Spectral::Quadrature::GaussLobatto);
    const auto& block = domain.blocks()[element_id.block_id()];
    Initialization::mutate_assign<
        tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>>(
        make_not_null(&box),
        inertial_coordinates(element_id, mesh, block, time, functions_of_time));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <size_t Dim>
struct TestResult {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& inertial_coords =
        get<domain::Tags::Coordinates<Dim, Frame::Inertial>>(box);
    const auto& scalar_field = get<ScalarFieldTag>(box);
    const auto expected_data = gaussian(inertial_coords);
    SPECTRE_PARALLEL_REQUIRE(
        equal_within_roundoff(get(scalar_field), expected_data, 1e-3));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using chare_type = Parallel::Algorithms::Array;
  using array_index = ElementId<Dim>;
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;

  using import_fields = tmpl::list<ScalarFieldTag, VectorFieldTag<Dim>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeElement<Dim>,
                                        Parallel::Actions::TerminatePhase>>,
      /// [import_actions]
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<importers::Actions::RegisterWithElementDataReader,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::ImportInitialData,
          tmpl::list<
              importers::Actions::ReadVolumeData<OptionsGroup, import_fields>,
              importers::Actions::ReceiveVolumeData<import_fields>,
              Parallel::Actions::TerminatePhase>>,
      /// [import_actions]
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<TestResult<Dim>, Parallel::Actions::TerminatePhase>>>;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
          initialization_items,
      const std::unordered_set<size_t>& procs_to_ignore = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& element_array =
        Parallel::get_parallel_component<ElementArray>(local_cache);
    const auto& domain =
        get<Tags::Domain<Dim, SourceOrTarget::Target>>(local_cache);
    const auto& initial_refinement_levels =
        get<Tags::InitialRefinementLevels<Dim, SourceOrTarget::Target>>(
            local_cache);
    const size_t num_procs = static_cast<size_t>(sys::number_of_procs());
    size_t which_proc = 0;
    for (const auto& block : domain.blocks()) {
      const std::vector<ElementId<Dim>> element_ids = initial_element_ids(
          block.id(), initial_refinement_levels[block.id()]);
      for (const auto& element_id : element_ids) {
        while (procs_to_ignore.find(which_proc) != procs_to_ignore.end()) {
          which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
        }
        element_array(element_id)
            .insert(global_cache, initialization_items, which_proc);
        which_proc = which_proc + 1 == num_procs ? 0 : which_proc + 1;
      }
    }
    element_array.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

/// [metavars]
template <size_t Dim>
struct Metavariables {
  static constexpr size_t volume_dim = Dim;
  struct system {};

  using component_list =
      tmpl::list<ElementArray<Dim, Metavariables>,
                 TestDataWriter<Dim, Metavariables>,
                 importers::ElementDataReader<Metavariables>>;

  static constexpr const char* const help{"Test the volume data reader"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Register,
       Parallel::Phase::ImportInitialData, Parallel::Phase::Testing,
       Parallel::Phase::Exit}};

  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<DomainCreator<Dim>, domain_creators<Dim>>>;
  };

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
/// [metavars]

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &setup_memory_allocation_failure_reporting,
    &domain::creators::register_derived_with_charm,
    &domain::creators::time_dependence::register_derived_with_charm,
    &domain::FunctionsOfTime::register_derived_with_charm,
    &register_factory_classes_with_charm<metavariables>};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};
