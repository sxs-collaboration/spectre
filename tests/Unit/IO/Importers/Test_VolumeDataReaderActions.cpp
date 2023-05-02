// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 2>;
  static std::string name() { return "V"; }
};

struct TensorTag : db::SimpleTag {
  using type = tnsr::ij<DataVector, 2>;
  static std::string name() { return "T"; }
};

using import_tags_list = tmpl::list<VectorTag, TensorTag>;

struct TestVolumeData {};

using ElementIdType = ElementId<2>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIdType;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::push_back<
                         import_tags_list,
                         domain::Tags::Coordinates<2, Frame::Inertial>>>,
                     importers::Actions::RegisterWithElementDataReader>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<importers::Actions::ReadVolumeData<TestVolumeData,
                                                        import_tags_list>,
                     importers::Actions::ReceiveVolumeData<import_tags_list>>>>;
};

template <typename Metavariables>
struct MockVolumeDataReader {
  using component_being_mocked = importers::ElementDataReader<Metavariables>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<importers::detail::InitializeElementDataReader<
          metavariables::volume_dim>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
};

void test_actions(const std::variant<double, importers::ObservationSelector>&
                      observation_selection) {
  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {importers::ImporterOptions{"TestVolumeData*.h5", "element_data",
                                  observation_selection, false}}};

  // Setup mock data file reader
  ActionTesting::emplace_nodegroup_component<reader_component>(
      make_not_null(&runner));
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<reader_component>(make_not_null(&runner), 0);
  }

  // Create a few elements with sample data
  // Specific IDs have no significance, just need different IDs.
  const std::vector<ElementId<2>> element_ids{{1, {{{1, 0}, {1, 0}}}},
                                              {1, {{{1, 1}, {1, 0}}}},
                                              {1, {{{1, 0}, {2, 3}}}},
                                              {1, {{{1, 0}, {5, 4}}}},
                                              {0, {{{1, 0}, {1, 0}}}}};
  std::unordered_map<ElementId<2>,
                     tuples::tagged_tuple_from_typelist<import_tags_list>>
      all_sample_data{};
  std::unordered_map<ElementId<2>, tnsr::I<DataVector, 2>> all_coords{};
  for (const auto& id : element_ids) {
    // Generate sample data
    auto& sample_data =
        all_sample_data
            .emplace(id, tuples::tagged_tuple_from_typelist<import_tags_list>{})
            .first->second;
    auto& coords = all_coords[id];
    const auto hashed_id = static_cast<double>(std::hash<ElementId<2>>{}(id));
    const size_t num_points = 4;
    get<0>(coords) = DataVector{1.5 * hashed_id, 2.3 * hashed_id,
                                3.4 * hashed_id, 5.6 * hashed_id};
    get<1>(coords) = DataVector{8.9 * hashed_id, 1.3 * hashed_id,
                                2.4 * hashed_id, 6.7 * hashed_id};
    tnsr::I<DataVector, 2> vector{num_points};
    get<0>(vector) = DataVector{0.5 * hashed_id, 1.0 * hashed_id,
                                3.0 * hashed_id, -2.0 * hashed_id};
    get<1>(vector) = DataVector{-0.5 * hashed_id, -1.0 * hashed_id,
                                -3.0 * hashed_id, 2.0 * hashed_id};
    get<VectorTag>(sample_data) = std::move(vector);
    tnsr::ij<DataVector, 2> tensor{num_points};
    get<0, 0>(tensor) = DataVector{10.5 * hashed_id, 11.0 * hashed_id,
                                   13.0 * hashed_id, -22.0 * hashed_id};
    get<0, 1>(tensor) = DataVector{10.5 * hashed_id, -11.0 * hashed_id,
                                   -13.0 * hashed_id, -22.0 * hashed_id};
    get<1, 0>(tensor) = DataVector{-10.5 * hashed_id, 11.0 * hashed_id,
                                   13.0 * hashed_id, 22.0 * hashed_id};
    get<1, 1>(tensor) = DataVector{-10.5 * hashed_id, -11.0 * hashed_id,
                                   -13.0 * hashed_id, 22.0 * hashed_id};
    get<TensorTag>(sample_data) = std::move(tensor);

    // Initialize element with no data, it should be populated from the volume
    // data file
    ActionTesting::emplace_component_and_initialize<element_array>(
        make_not_null(&runner), ElementIdType{id},
        {tnsr::I<DataVector, 2>{}, tnsr::ij<DataVector, 2>{}, coords});

    // Register element
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    // Invoke the simple_action RegisterElementWithSelf that was called on the
    // reader component by the RegisterWithElementDataReader action.
    runner.invoke_queued_simple_action<reader_component>(0);
  }

  const auto get_element_tag =
      [&runner](auto tag_v, const ElementId<2>& local_id) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, local_id);
  };
  const auto get_reader_tag = [&runner](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<reader_component, tag>(runner, 0);
  };

  // Collect the sample data from all elements.
  // The outer vector represents a list of files that we split the element data
  // into, so we can test loading volume data from multiple files.
  std::vector<std::vector<ElementVolumeData>> all_element_data(2);
  for (const auto& id : element_ids) {
    const std::string element_name = MakeString{} << id;
    std::vector<TensorComponent> tensor_data(8);
    const auto& vector = get<VectorTag>(all_sample_data.at(id));
    tensor_data[0] = TensorComponent("V_x"s, get<0>(vector));
    tensor_data[1] = TensorComponent("V_y"s, get<1>(vector));
    const auto& tensor = get<TensorTag>(all_sample_data.at(id));
    tensor_data[2] = TensorComponent("T_xx"s, get<0, 0>(tensor));
    tensor_data[3] = TensorComponent("T_xy"s, get<0, 1>(tensor));
    tensor_data[4] = TensorComponent("T_yx"s, get<1, 0>(tensor));
    tensor_data[5] = TensorComponent("T_yy"s, get<1, 1>(tensor));
    const auto& coords = all_coords.at(id);
    tensor_data[6] = TensorComponent("InertialCoordinates_x"s, get<0>(coords));
    tensor_data[7] = TensorComponent("InertialCoordinates_y"s, get<1>(coords));
    all_element_data[id.block_id()].push_back(
        {element_name,
         tensor_data,
         {2, 2},
         {2, Spectral::Basis::Legendre},
         {2, Spectral::Quadrature::GaussLobatto}});
  }
  // Write the sample data into an H5 file
  for (size_t i = 0; i < all_element_data.size(); ++i) {
    const std::string h5_file_name =
        "TestVolumeData" + std::to_string(i) + ".h5";
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
    h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name, false};
    auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
    volume_data.write_volume_data(0, 0., all_element_data[i]);
  }

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  bool first_invocation = true;
  for (const auto& id : element_ids) {
    // `ReadVolumeData`
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    if (first_invocation) {
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<element_array>(
          make_not_null(&runner), id));
    }
    CHECK(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}).size() ==
          (first_invocation ? 0 : 1));
    // Invoke the simple_action `ReadAllVolumeDataAndDistribute` that was called
    // on the reader component by the `ReadVolumeData` action.
    runner.invoke_queued_simple_action<reader_component>(0);
    CAPTURE(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}));
    CHECK(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}).size() ==
          1);
    // `ReceiveVolumeData` should be ready now
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    // Check the received data
    CHECK(get_element_tag(VectorTag{}, id) ==
          get<VectorTag>(all_sample_data.at(id)));
    CHECK(get_element_tag(TensorTag{}, id) ==
          get<TensorTag>(all_sample_data.at(id)));
    first_invocation = false;
  }

  for (size_t i = 0; i < all_element_data.size(); ++i) {
    const std::string h5_file_name =
        "TestVolumeData" + std::to_string(i) + ".h5";
    if (file_system::check_if_file_exists(h5_file_name)) {
      file_system::rm(h5_file_name, true);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.VolumeDataReaderActions", "[Unit][IO]") {
  test_actions(0.);
  test_actions(importers::ObservationSelector::First);
  test_actions(importers::ObservationSelector::Last);
}
