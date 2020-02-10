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
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/ElementActions.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Importers/VolumeDataReader.hpp"
#include "IO/Importers/VolumeDataReaderActions.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

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

struct TestVolumeData {
  using group = importers::OptionTags::Group;
};

using ElementIndexType = ElementIndex<2>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndexType;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<import_tags_list>,
                 importers::Actions::RegisterWithVolumeDataReader>>>;
};

template <typename Metavariables>
struct MockVolumeDataReader {
  using component_being_mocked = importers::VolumeDataReader<Metavariables>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,
      tmpl::list<importers::detail::InitializeVolumeDataReader>>>;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<importers::Tags::FileName<TestVolumeData>,
                 importers::Tags::Subgroup<TestVolumeData>,
                 importers::Tags::ObservationValue<TestVolumeData>>;
  enum class Phase { Initialization, Testing };
};

struct TestCallback {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename DataBox = db::DataBox<DbTagsList>,
            Requires<db::tag_is_retrievable_v<VectorTag, DataBox>> = nullptr>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndexType& /*array_index*/,
                    tuples::tagged_tuple_from_typelist<import_tags_list>
                        tensor_data) noexcept {
    CHECK(get<VectorTag>(tensor_data) == get<VectorTag>(box));
    CHECK(get<TensorTag>(tensor_data) == get<TensorTag>(box));
  }
};

}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.VolumeDataReaderActions", "[Unit][IO]") {
  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {"TestVolumeData.h5", "element_data", 0.}};

  // Setup mock data file reader
  ActionTesting::emplace_component<reader_component>(make_not_null(&runner), 0);
  ActionTesting::next_action<reader_component>(make_not_null(&runner), 0);

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
    ActionTesting::emplace_component_and_initialize<element_array>(
        make_not_null(&runner), ElementIndexType{id},
        {std::move(vector), std::move(tensor)});

    // Register element
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterElementWithSelf that was called on the
    // reader component by the RegisterWithVolumeDataReader action.
    runner.invoke_queued_simple_action<reader_component>(0);
  }

  const auto get_element_tag =
      [&runner](auto tag_v, const ElementId<2>& local_id) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, local_id);
  };

  // Collect the sample data from all elements
  std::vector<ExtentsAndTensorVolumeData> all_element_data{};
  for (const auto& id : element_ids) {
    const std::string element_name = MakeString{} << id << '/';
    std::vector<TensorComponent> tensor_data(6);
    const auto& vector = get_element_tag(VectorTag{}, id);
    tensor_data[0] = TensorComponent(element_name + "V_x"s, get<0>(vector));
    tensor_data[1] = TensorComponent(element_name + "V_y"s, get<1>(vector));
    const auto& tensor = get_element_tag(TensorTag{}, id);
    tensor_data[2] = TensorComponent(element_name + "T_xx"s, get<0, 0>(tensor));
    tensor_data[3] = TensorComponent(element_name + "T_xy"s, get<0, 1>(tensor));
    tensor_data[4] = TensorComponent(element_name + "T_yx"s, get<1, 0>(tensor));
    tensor_data[5] = TensorComponent(element_name + "T_yy"s, get<1, 1>(tensor));
    all_element_data.push_back({{2, 2}, tensor_data});
  }
  // Write the sample data into an H5 file
  const std::string h5_file_name = "TestVolumeData.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name, false};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  volume_data.write_volume_data(0, 0., all_element_data);

  runner.set_phase(Metavariables::Phase::Testing);

  // Have the importer read the file and pass it to the callback
  runner.algorithms<reader_component>()
      .at(0)
      .template threaded_action<importers::ThreadedActions::ReadVolumeData<
          TestVolumeData, import_tags_list, TestCallback, element_array>>();
  runner.invoke_queued_threaded_action<reader_component>(0);

  // Invoke the queued callbacks on the elements that test if the data is
  // correct
  for (const auto& id : element_ids) {
    runner.invoke_queued_simple_action<element_array>(id);
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
