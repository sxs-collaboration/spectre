// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "IO/DataImporter/DataFileReader.hpp"
#include "IO/DataImporter/DataFileReaderActions.hpp"
#include "IO/DataImporter/ElementActions.hpp"
#include "IO/DataImporter/Tags.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

#include "Parallel/Printf.hpp"

namespace {

struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 2>;
  static std::string name() noexcept { return "V"; }
};

struct TensorTag : db::SimpleTag {
  using type = tnsr::ij<DataVector, 2>;
  static std::string name() noexcept { return "T"; }
};

using import_tags_list = tmpl::list<VectorTag, TensorTag>;

struct TestImporter {
  using group = importer::OptionTags::Group;
  static constexpr OptionString help = "halp";
};

using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using initial_databox = db::DataBox<import_tags_list>;
};

template <typename Metavariables>
struct MockDataFileReader {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list = tmpl::list<>;
  using component_being_mocked = importer::DataFileReader<Metavariables>;
  using initial_databox =
      typename importer::DataFileReader<Metavariables>::initial_databox;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockDataFileReader<Metavariables>>;
  using const_global_cache_tag_list =
      tmpl::list<importer::OptionTags::DataFileName<TestImporter>,
                 importer::OptionTags::VolumeDataSubgroup<TestImporter>,
                 importer::OptionTags::ObservationValue<TestImporter>>;
  enum class Phase {};
};

struct TestCallback {
  template <typename... DbTags, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<tmpl::list<DbTags...>>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndexType& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/,
                    tuples::tagged_tuple_from_typelist<import_tags_list>
                        tensor_data) noexcept {
    CHECK(get<VectorTag>(tensor_data) == get<VectorTag>(box));
    CHECK(get<TensorTag>(tensor_data) == get<TensorTag>(box));
  }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.IO.DataImporter.DataFileReaderActions", "[Unit][IO]") {
  using TupleOfMockDistributedObjects =
      typename ActionTesting::MockRuntimeSystem<
          Metavariables>::TupleOfMockDistributedObjects;
  using reader_component = MockDataFileReader<Metavariables>;
  using array_component = MockElementArray<Metavariables>;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using ReaderMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          reader_component>;
  using ArrayMockDistributedObjectsTag =
      typename MockRuntimeSystem::template MockDistributedObjectsTag<
          array_component>;
  TupleOfMockDistributedObjects dist_objects{};
  tuples::get<ReaderMockDistributedObjectsTag>(dist_objects)
      .emplace(0, ActionTesting::MockDistributedObject<reader_component>{});

  // Create a few elements with sample data
  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  for (const auto& id : element_ids) {
    const auto hashed_id = static_cast<double>(std::hash<ElementId<2>>{}(id));
    const size_t num_points = 4;
    db::item_type<VectorTag> vector{num_points};
    get<0>(vector) = DataVector{0.5 * hashed_id, 1.0 * hashed_id,
                                3.0 * hashed_id, -2.0 * hashed_id};
    get<1>(vector) = DataVector{-0.5 * hashed_id, -1.0 * hashed_id,
                                -3.0 * hashed_id, 2.0 * hashed_id};
    db::item_type<TensorTag> tensor{num_points};
    get<0, 0>(tensor) = DataVector{10.5 * hashed_id, 11.0 * hashed_id,
                                   13.0 * hashed_id, -22.0 * hashed_id};
    get<0, 1>(tensor) = DataVector{10.5 * hashed_id, -11.0 * hashed_id,
                                   -13.0 * hashed_id, -22.0 * hashed_id};
    get<1, 0>(tensor) = DataVector{-10.5 * hashed_id, 11.0 * hashed_id,
                                   13.0 * hashed_id, 22.0 * hashed_id};
    get<1, 1>(tensor) = DataVector{-10.5 * hashed_id, -11.0 * hashed_id,
                                   -13.0 * hashed_id, 22.0 * hashed_id};
    tuples::get<ArrayMockDistributedObjectsTag>(dist_objects)
        .emplace(
            ElementIndex<2>{id},
            db::create<import_tags_list>(std::move(vector), std::move(tensor)));
  }

  // Setup the runner with its cache
  tuples::TaggedTuple<importer::OptionTags::DataFileName<TestImporter>,
                      importer::OptionTags::VolumeDataSubgroup<TestImporter>,
                      importer::OptionTags::ObservationValue<TestImporter>>
      cache_data{"TestImportVolumeFile.h5", "element_data", 0.};
  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      cache_data, std::move(dist_objects)};

  const auto get_element_box =
      [&runner](const ElementId<2>& element_id) -> decltype(auto) {
    return runner.algorithms<array_component>()
        .at(element_id)
        .get_databox<array_component::initial_databox>();
  };

  runner.simple_action<reader_component,
                       importer::detail::InitializeDataFileReader>(0);

  // Register elements
  for (const auto& id : element_ids) {
    runner.simple_action<array_component,
                         importer::Actions::RegisterWithImporter>(id);
    // Invoke the simple_action RegisterElementWithSelf that was called on the
    // reader component by the RegisterWithImporter action.
    runner.invoke_queued_simple_action<reader_component>(0);
  }

  // Write the sample data into an H5 file
  const std::string h5_file_name = "TestImportVolumeFile.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name, false};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  for (const auto& id : element_ids) {
    const auto& element_box = get_element_box(id);
    const std::string element_name = MakeString{} << id << '/';
    std::vector<TensorComponent> tensor_data(6);
    const auto& vector = get<VectorTag>(element_box);
    tensor_data[0] = TensorComponent(element_name + "V_x"s, get<0>(vector));
    tensor_data[1] = TensorComponent(element_name + "V_y"s, get<1>(vector));
    const auto& tensor = get<TensorTag>(element_box);
    tensor_data[2] = TensorComponent(element_name + "T_xx"s, get<0, 0>(tensor));
    tensor_data[3] = TensorComponent(element_name + "T_xy"s, get<0, 1>(tensor));
    tensor_data[4] = TensorComponent(element_name + "T_yx"s, get<1, 0>(tensor));
    tensor_data[5] = TensorComponent(element_name + "T_yy"s, get<1, 1>(tensor));
    volume_data.insert_tensor_data(
        0, 0., ExtentsAndTensorVolumeData{{2, 2}, tensor_data});
  }

  Parallel::printf("Queue read data action\n");
  // Have the importer read the file and pass it to the callback
  runner.algorithms<reader_component>()
      .at(0)
      .template threaded_action<importer::ThreadedActions::ReadElementData<
          TestImporter, import_tags_list, TestCallback, array_component>>();
  runner.invoke_queued_threaded_action<reader_component>(0);

  // Invoke the queued callbacks on the elements that test if the data is
  // correct
  for (const auto& id : element_ids) {
    runner.invoke_queued_simple_action<array_component>(id);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
