// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <pup.h>
#include <string>
#include <tuple>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Evolution/Systems/Cce/Events/ObserveFields.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/Dat.hpp"
#include "IO/H5/File.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "IO/Observer/Tags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Event.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace {
using ObserveFields = Events::ObserveFields;
constexpr size_t l_max = 3;
constexpr size_t num_radial_grid_points = 2;
constexpr size_t num_angular_grid_points =
    Spectral::Swsh::number_of_swsh_collocation_points(l_max);
constexpr size_t num_volume_grid_points =
    num_radial_grid_points * num_angular_grid_points;
constexpr double time = 1.3;

struct MockWriteVolumeData {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex,
            typename DataBox = db::DataBox<DbTagsList>>
  static void apply(db::DataBox<DbTagsList>& /*box*/,
                    Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const gsl::not_null<Parallel::NodeLock*> /*node_lock*/,
                    const std::string& /*h5_file_name*/,
                    const std::string& subfile_path,
                    const observers::ObservationId& /*observation_id*/,
                    std::vector<ElementVolumeData>&& volume_data) {
    // If we were to have made the data non-zero values, the event does too many
    // transformations for us to calculate what the result would be by hand, so
    // we just check that the sizes are correct, and that the (code) time is
    // correct
    const size_t l_plus_one_squared = square(l_max + 1);
    const DataVector& data_v =
        std::get<DataVector>(volume_data[0].tensor_components[0].data);
    if (subfile_path.find("InertialRetardedTime") != std::string::npos) {
      CHECK(data_v.size() == 2 * l_plus_one_squared);
    } else if (subfile_path.find("OneMinusY") != std::string::npos) {
      CHECK(data_v.size() == num_radial_grid_points);
    } else {
      CHECK(data_v.size() ==
            2 * l_plus_one_squared * num_radial_grid_points);
    }
  }
};

void check_h5_file(const std::string& filename_prefix) {
  const h5::H5File<h5::AccessType::ReadOnly> h5_file{filename_prefix + ".h5"};
  const size_t l_plus_one_squared = square(l_max + 1);
  {
    const auto& u_volume_data =
        h5_file.get<h5::VolumeData>("/CceVolumeData/InertialRetardedTime");
    const size_t observation_id = u_volume_data.find_observation_id(time);

    const TensorComponent inertial_retarded_time_component =
        u_volume_data.get_tensor_component(observation_id,
                                           "InertialRetardedTime");

    const DataVector inertial_retarded_time =
        std::get<DataVector>(inertial_retarded_time_component.data);

    CHECK(inertial_retarded_time.size() == 2 * l_plus_one_squared);
    h5_file.close_current_object();
  }
  {
    const auto& one_minus_y_volume_data =
        h5_file.get<h5::VolumeData>("/CceVolumeData/OneMinusY");
    const size_t observation_id =
        one_minus_y_volume_data.find_observation_id(time);

    const TensorComponent one_minus_y_component =
        one_minus_y_volume_data.get_tensor_component(observation_id,
                                           "OneMinusY");

    const DataVector one_minus_y =
        std::get<DataVector>(one_minus_y_component.data);

    CHECK(one_minus_y.size() == num_radial_grid_points);
    h5_file.close_current_object();
  }
  {
    const auto& volume_data =
        h5_file.get<h5::VolumeData>("/CceVolumeData/VolumeData");
    const size_t observation_id = volume_data.find_observation_id(time);

    const TensorComponent J_component =
        volume_data.get_tensor_component(observation_id, "J");

    const DataVector J_interleaved =
        std::get<DataVector>(J_component.data);
    CHECK(J_interleaved.size() ==
          2 * l_plus_one_squared * num_radial_grid_points);
    h5_file.close_current_object();
  }
}

template <typename Metavariables>
struct MockObserverWriter {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockNodeGroupChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 tmpl::list<observers::Tags::H5FileLock>>>>>;
  using const_global_cache_tags =
      tmpl::list<observers::Tags::VolumeFileName>;
  using component_being_mocked = observers::ObserverWriter<Metavariables>;

  using replace_these_threaded_actions =
      tmpl::list<observers::ThreadedActions::WriteVolumeData>;
  using with_these_threaded_actions = tmpl::list<MockWriteVolumeData>;
};

template <typename Metavariables>
struct MockElement {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockSingletonChare;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<tmpl::push_back<
          tmpl::list_difference<
            ObserveFields::available_tags_to_observe,
            tmpl::list<
              Tags::Psi0, Tags::Psi1, Tags::Psi2,
              Tags::NewmanPenroseAlpha, Tags::NewmanPenroseBeta,
              Tags::NewmanPenroseGamma, Tags::NewmanPenroseEpsilon,
              // Tags::NewmanPenroseKappa, // in our tetrad, \kappa=0
              Tags::NewmanPenroseTau, Tags::NewmanPenroseSigma,
              Tags::NewmanPenroseRho,
              Tags::NewmanPenrosePi, Tags::NewmanPenroseNu,
              Tags::NewmanPenroseMu, Tags::NewmanPenroseLambda
              >>,
          Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                           Spectral::Swsh::Tags::Eth>,
          Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                           Spectral::Swsh::Tags::Eth>,
          Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                           Spectral::Swsh::Tags::Ethbar>,
          Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                           Spectral::Swsh::Tags::Eth>,
          Tags::Exp2Beta,
          Tags::BondiK, Tags::LMax, Tags::NumberOfRadialPoints,
          ::Tags::Time>>>>>;
};

struct Metavars {
  void pup(PUP::er& /*p*/) {}
  using observed_reduction_data_tags = tmpl::list<>;
  using component_list =
      tmpl::list<MockObserverWriter<Metavars>, MockElement<Metavars>>;
};

void test(const bool write_synchronously) {
  using metavars = Metavars;
  using obs_writer = MockObserverWriter<metavars>;
  using element = MockElement<metavars>;

  const std::vector<std::string> observe_field_names{
    "InertialRetardedTime", "J", "Psi0", "Psi1", "Psi2", "Dy(H)", "OneMinusY",
    "NewmanPenroseAlpha"};
  const std::string subgroup_name{"CceVolumeData"};
  const Cce::Events::ObserveFields fields{subgroup_name, observe_field_names};
  const Cce::Events::ObserveFields serialized_fields =
      serialize_and_deserialize(fields);

  const std::string volume_filename_prefix{"PineTree"};
  if (file_system::check_if_file_exists(volume_filename_prefix + ".h5")) {
    file_system::rm(volume_filename_prefix + ".h5", true);
  }

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  MockRuntimeSystem runner{
      {volume_filename_prefix},
      {},
      std::vector<size_t>{write_synchronously ? 1_st : 2_st}};
  ActionTesting::emplace_nodegroup_component_and_initialize<obs_writer>(
      make_not_null(&runner), {Parallel::NodeLock{}});
  ActionTesting::emplace_singleton_component_and_initialize<element>(
      make_not_null(&runner), ActionTesting::NodeId{0},
      ActionTesting::LocalCoreId{0}, {});

  auto& cache = ActionTesting::cache<obs_writer>(runner, 0);
  auto& box = ActionTesting::get_databox<element>(make_not_null(&runner), 0);
  auto obs_box = make_observation_box<
      typename Cce::Events::ObserveFields::compute_tags_for_observation_box>(
      make_not_null(&box));
  const int array_index = 0;
  const obs_writer* const component = nullptr;
  const Event::ObservationValue observation_value{};

  // We only set the sizes of the tensors that we are using
  const auto set_number = [&box](const auto tag_v, const auto value_to_set_to) {
    using tag = std::decay_t<decltype(tag_v)>;
    db::mutate<tag>(
        [&value_to_set_to](
            const gsl::not_null<typename tag::type*> value_to_set) {
          *value_to_set = value_to_set_to;
        },
        make_not_null(&box));
  };
  const auto size_data = [&box](auto tag_v, const size_t size) {
    using tag = std::decay_t<decltype(tag_v)>;
    db::mutate<tag>(
        [&size](const auto tensor) {
          // All are scalars
          get(*tensor) = ComplexDataVector{size, 1.0};
        },
        make_not_null(&box));
  };
  set_number(Tags::LMax{}, l_max);
  set_number(Tags::NumberOfRadialPoints{}, num_radial_grid_points);
  set_number(::Tags::Time{}, time);
  size_data(Tags::ComplexInertialRetardedTime{}, num_angular_grid_points);

  tmpl::for_each<
      tmpl::list<Tags::BondiJ, Tags::OneMinusY,
                 Tags::Dy<Tags::BondiJ>, Tags::Dy<Tags::Dy<Tags::BondiJ>>,
                 Tags::BondiK, Tags::BondiQ, Tags::Dy<Tags::BondiQ>,
                 Tags::EthRDividedByR,
                 Tags::BondiBeta, Tags::Dy<Tags::BondiBeta>,
                 Tags::BondiH, Tags::Dy<Tags::BondiH>,
                 Tags::BondiU, Tags::Dy<Tags::BondiU>,
                 Tags::BondiW, Tags::Dy<Tags::BondiW>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiBeta,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::Dy<Tags::BondiBeta>,
                                                  Spectral::Swsh::Tags::Eth>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiJ,
                                                  Spectral::Swsh::Tags::Ethbar>,
                 Spectral::Swsh::Tags::Derivative<Tags::BondiU,
                                                  Spectral::Swsh::Tags::Eth>,
                 Tags::Exp2Beta,
                 Tags::BondiR>>([&size_data](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    size_data(tag{}, num_volume_grid_points);
  });

  serialized_fields(obs_box, cache, array_index, component, observation_value);

  if (write_synchronously) {
    check_h5_file(volume_filename_prefix);
  } else {
    // 1 each for InertialRetardedTime, OneMinusY, and 1 for all the rest of the
    // volume fields together.
    const size_t expected_number_of_actions = 3;
    CHECK(ActionTesting::number_of_queued_threaded_actions<obs_writer>(
              runner, 0) == expected_number_of_actions);
    for (size_t i = 0; i < expected_number_of_actions; i++) {
      ActionTesting::invoke_queued_threaded_action<obs_writer>(
          make_not_null(&runner), 0);
    }
  }
}

void test_errors() {
  CHECK_THROWS_WITH((Cce::Events::ObserveFields{"CceVolumeData",
        std::vector<std::string>{"Unique", "Duplicate", "Duplicate"}}),
                    Catch::Matchers::ContainsSubstring(
                        "more than once in list of variables to observe"));

  CHECK_THROWS_WITH(
      (Cce::Events::ObserveFields{"CceVolumeData",
         std::vector<std::string>{"MisspelledTag"}}),
      Catch::Matchers::ContainsSubstring(
          "MisspelledTag is not an available variable. Available variables:"));
}

SPECTRE_TEST_CASE("Unit.Evolution.Systems.Cce.Events.ObserveFields",
                  "[Unit][Evolution]") {
  test(true);
  test(false);
  test_errors();
}
}  // namespace
}  // namespace Cce
