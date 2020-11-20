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
#include "DataStructures/Tensor/TensorData.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/ActionTesting.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/VolumeData.hpp"
#include "IO/Importers/Actions/ReadVolumeData.hpp"
#include "IO/Importers/Actions/ReceiveVolumeData.hpp"
#include "IO/Importers/Actions/RegisterWithElementDataReader.hpp"
#include "IO/Importers/ElementDataReader.hpp"
#include "IO/Importers/Tags.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/ArrayIndex.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/TaggedTuple.hpp"

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

using ElementIdType = ElementId<2>;

template <typename Metavariables>
struct MockElementArray {
  using component_being_mocked = void;  // Not needed
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIdType;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<import_tags_list>,
                     importers::Actions::RegisterWithElementDataReader>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<importers::Actions::ReadVolumeData<TestVolumeData,
                                                        import_tags_list>,
                     importers::Actions::ReceiveVolumeData<TestVolumeData,
                                                           import_tags_list>>>>;
};

template <typename Metavariables>
struct MockVolumeDataReader {
  using component_being_mocked = importers::ElementDataReader<Metavariables>;
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename Metavariables::Phase, Metavariables::Phase::Initialization,

      tmpl::list<Actions::SetupDataBox,
                 importers::detail::InitializeElementDataReader>>>;
};

struct Metavariables {
  using component_list = tmpl::list<MockElementArray<Metavariables>,
                                    MockVolumeDataReader<Metavariables>>;
  enum class Phase { Initialization, Testing };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.IO.Importers.VolumeDataReaderActions", "[Unit][IO]") {
  using reader_component = MockVolumeDataReader<Metavariables>;
  using element_array = MockElementArray<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      {"TestVolumeData.h5", "element_data", 0.}};

  // Setup mock data file reader
  ActionTesting::emplace_component<reader_component>(make_not_null(&runner), 0);
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
  for (const auto& id : element_ids) {
    // Generate sample data
    auto& sample_data =
        all_sample_data
            .emplace(id, tuples::tagged_tuple_from_typelist<import_tags_list>{})
            .first->second;
    const auto hashed_id = static_cast<double>(std::hash<ElementId<2>>{}(id));
    const size_t num_points = 4;
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
        {tnsr::I<DataVector, 2>{}, tnsr::ij<DataVector, 2>{}});

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

  // Collect the sample data from all elements
  std::vector<ElementVolumeData> all_element_data{};
  for (const auto& id : element_ids) {
    const std::string element_name = MakeString{} << id << '/';
    std::vector<TensorComponent> tensor_data(6);
    const auto& vector = get<VectorTag>(all_sample_data.at(id));
    tensor_data[0] = TensorComponent(element_name + "V_x"s, get<0>(vector));
    tensor_data[1] = TensorComponent(element_name + "V_y"s, get<1>(vector));
    const auto& tensor = get<TensorTag>(all_sample_data.at(id));
    tensor_data[2] = TensorComponent(element_name + "T_xx"s, get<0, 0>(tensor));
    tensor_data[3] = TensorComponent(element_name + "T_xy"s, get<0, 1>(tensor));
    tensor_data[4] = TensorComponent(element_name + "T_yx"s, get<1, 0>(tensor));
    tensor_data[5] = TensorComponent(element_name + "T_yy"s, get<1, 1>(tensor));
    all_element_data.push_back({{2, 2},
                                tensor_data,
                                {2, Spectral::Basis::Legendre},
                                {2, Spectral::Quadrature::GaussLobatto}});
  }
  // Write the sample data into an H5 file
  const std::string h5_file_name = "TestVolumeData.h5";
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name, false};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  volume_data.write_volume_data(0, 0., all_element_data);

  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);

  bool first_invocation = true;
  for (const auto& id : element_ids) {
    // `ReadVolumeData`
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    // `ReceiveVolumeData` should not be ready on the first invocation, since
    // no data has been read yet.
    CHECK(ActionTesting::is_ready<element_array>(runner, id) !=
          first_invocation);
    CHECK(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}).size() ==
          (first_invocation ? 0 : 1));
    // Invoke the simple_action `ReadAllVolumeDataAndDistribute` that was called
    // on the reader component by the `ReadVolumeData` action.
    runner.invoke_queued_simple_action<reader_component>(0);
    CAPTURE(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}));
    CHECK(get_reader_tag(importers::Tags::ElementDataAlreadyRead{}).size() ==
          1);
    // `ReceiveVolumeData` should be ready now
    CHECK(ActionTesting::is_ready<element_array>(runner, id));
    ActionTesting::next_action<element_array>(make_not_null(&runner), id);
    // Check the received data
    CHECK(get_element_tag(VectorTag{}, id) ==
          get<VectorTag>(all_sample_data.at(id)));
    CHECK(get_element_tag(TensorTag{}, id) ==
          get<TensorTag>(all_sample_data.at(id)));
    first_invocation = false;
  }

  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
}
