// Distributed under the MIT License.
// See LICENSE.txt for details.

#define CATCH_CONFIG_RUNNER

#include "Framework/TestingFramework.hpp"

#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "Helpers/Parallel/RoundRobinArrayElements.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Main.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Initialization/MutateAssign.hpp"
#include "Time/Slab.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/System/ParallelInfo.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

static constexpr int total_number_of_array_elements = 12;

namespace PUP {
class er;
}  // namespace PUP
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db

namespace {
template <size_t Dim>
using BoundaryMessage = evolution::dg::BoundaryMessage<Dim>;

int node_of(const int element, const bool using_two_nodes) {
  return using_two_nodes
             ? (element < total_number_of_array_elements / 2 ? 0 : 1)
             : 0;
}

namespace Tags {
// When we are running with two charm nodes, we want some elements to send to
// the same node and some to send to a different node. Here's a chart:
//
// Element  MyNode  SendingTo  TheirNode  Inter/Intra-Communication
//    0       0         1          0         Intra
//    1       0         2          0         Intra
//    2       0         3          0         Intra
//    3       0         6          1         Inter
//    4       0        11          1         Inter
//    5       0        10          1         Inter
//    6       1         7          1         Intra
//    7       1         8          1         Intra
//    8       1         9          1         Intra
//    9       1         0          0         Inter
//   10       1         4          0         Inter
//   11       1         5          0         Inter
//
// Six elements will be doing inter-node communication, and six will be doing
// intra-node communication
//
// When we are running with only one charm node, all twelve elements are on the
// same node and will be doing intra-node communication
struct SendMap : db::SimpleTag {
  using type = std::map<int, int>;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options() {
    return std::map<int, int>{{0, 1}, {1, 2}, {2, 3}, {3, 6}, {4, 11}, {5, 10},
                              {6, 7}, {7, 8}, {8, 9}, {9, 0}, {10, 4}, {11, 5}};
  }
};

// Inverse of SendMap. Simple way to have a bidirectional mapping
struct ReceiveMap : db::SimpleTag {
  using type = std::map<int, int>;
  using option_tags = tmpl::list<>;
  static constexpr bool pass_metavariables = false;

  static type create_from_options() {
    const std::map<int, int> send_map = SendMap::create_from_options();

    std::map<int, int> receive_map{};

    for (const auto& [sender_element, receiver_element] : send_map) {
      receive_map[receiver_element] = sender_element;
    }

    return receive_map;
  }
};

// We want to be able to compare Vector1.data() on the receiving element to the
// Vector0.data() of the sending element. If the two elements are on the same
// node, then those two addresses should be the same. If they are on different
// nodes then they should be different. This is just a way to communicate
// Vector0.data() from the sender to the receiver to compare.
struct AddressOfVector0OnSender : db::SimpleTag {
  using type = std::string;
};

// Even though some of these can be accessed with functions, we add them all
// here to have a uniform interface
// {
struct MyElement : db::SimpleTag {
  using type = int;
};

struct MyNode : db::SimpleTag {
  using type = int;
};

struct ElementToSendTo : db::SimpleTag {
  using type = int;
};

struct NodeOfElementToSendTo : db::SimpleTag {
  using type = int;
};

struct ElementToReceiveFrom : db::SimpleTag {
  using type = int;
};

struct NodeOfElementToReceiveFrom : db::SimpleTag {
  using type = int;
};
// }

struct Vector0 : db::SimpleTag {
  using type = DataVector;
};

struct Vector1 : db::SimpleTag {
  using type = DataVector;
};

// We hijack the tci_status argument to be the array index of the receiving
// element. Also this comment is up here so the docs look pretty
// [charm message inbox tag]
struct BoundaryMessageReceiveTag {
  using temporal_id = int;
  using type =
      std::unordered_map<temporal_id, std::unique_ptr<BoundaryMessage<3>>>;
  using message_type = BoundaryMessage<3>;

  template <typename Inbox>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                BoundaryMessage<3>* message) {
    const int receiver_element = message->tci_status;
    (*inbox)[receiver_element] = std::unique_ptr<BoundaryMessage<3>>(message);
  }
};
// [charm message inbox tag]
}  // namespace Tags

struct SendAddressOfVector0 {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const std::string& sent_address) {
    db::mutate<Tags::AddressOfVector0OnSender>(
        [&sent_address](const gsl::not_null<std::string*> address) {
          *address = sent_address;
        },
        make_not_null(&box));
  }
};

struct Initialize {
  using simple_tags =
      tmpl::list<Tags::MyElement, Tags::MyNode, Tags::ElementToSendTo,
                 Tags::NodeOfElementToSendTo, Tags::ElementToReceiveFrom,
                 Tags::NodeOfElementToReceiveFrom, Tags::Vector0, Tags::Vector1,
                 Tags::AddressOfVector0OnSender>;
  using const_global_cache_tags = tmpl::list<Tags::SendMap, Tags::ReceiveMap>;
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& send_map = Parallel::get<Tags::SendMap>(cache);
    const auto& receive_map = Parallel::get<Tags::ReceiveMap>(cache);
    const int element_to_send_to = send_map.at(array_index);
    const int element_to_receive_from = receive_map.at(array_index);
    const bool using_two_nodes = Parallel::number_of_nodes<int>(cache) == 2;

    Initialization::mutate_assign<
        tmpl::list<Tags::MyElement, Tags::MyNode, Tags::ElementToSendTo,
                   Tags::NodeOfElementToSendTo, Tags::ElementToReceiveFrom,
                   Tags::NodeOfElementToReceiveFrom, Tags::Vector0>>(
        make_not_null(&box), array_index, node_of(array_index, using_two_nodes),
        element_to_send_to, node_of(element_to_send_to, using_two_nodes),
        element_to_receive_from,
        node_of(element_to_receive_from, using_two_nodes),
        DataVector{static_cast<size_t>(array_index) + 1, 1.0});

    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct SendMessage {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const int my_node = db::get<Tags::MyNode>(box);
    const int element_to_send_to = db::get<Tags::ElementToSendTo>(box);

    auto& proxy = Parallel::get_parallel_component<ParallelComponent>(cache);

    // We hijack the tci_status argument to be the array index of the element
    // that we are sending data to which will be the temporal id for the inbox
    // tag
    BoundaryMessage<3>* message = new BoundaryMessage<3>(
        0, static_cast<size_t>(array_index + 1), false, true,
        static_cast<size_t>(my_node), static_cast<size_t>(my_node),
        element_to_send_to, {}, {}, {}, {}, {}, {}, nullptr,
        const_cast<double*>(db::get<Tags::Vector0>(box).data()));

    std::stringstream ss{};
    ss << message;
    const std::string message_address{ss.str()};
    ss.str("");
    ss << message->dg_flux_data;
    const std::string data_address{ss.str()};

    Parallel::receive_data<Tags::BoundaryMessageReceiveTag>(
        proxy[element_to_send_to], message);

    // Send the address of Vector0.data() to the receiver
    Parallel::simple_action<SendAddressOfVector0>(proxy[element_to_send_to],
                                                  data_address);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct ReceiveMessage {
  using inbox_tags = tmpl::list<Tags::BoundaryMessageReceiveTag>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = tuples::get<Tags::BoundaryMessageReceiveTag>(inboxes);

    if (inbox.count(array_index) == 0) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    // Scope this so we don't get a dangling reference
    {
      auto& boundary_message_ptr = inbox.at(array_index);

      db::mutate<Tags::Vector1>(
          [&boundary_message_ptr](const gsl::not_null<DataVector*> vector_1) {
            vector_1->set_data_ref(boundary_message_ptr->dg_flux_data,
                                   boundary_message_ptr->dg_flux_data_size);
          },
          make_not_null(&box));

      const int my_node = db::get<Tags::MyNode>(box);
      const int node_of_element_to_receive_from =
          db::get<Tags::NodeOfElementToReceiveFrom>(box);

      if (node_of_element_to_receive_from == my_node) {
        SPECTRE_PARALLEL_REQUIRE_FALSE(boundary_message_ptr->owning);
      } else {
        SPECTRE_PARALLEL_REQUIRE(boundary_message_ptr->owning);
      }
    }

    inbox.erase(array_index);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

struct CheckMessage {
  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const int my_element = db::get<Tags::MyElement>(box);
    const int my_node = db::get<Tags::MyNode>(box);
    const int element_to_receive_from =
        db::get<Tags::ElementToReceiveFrom>(box);
    const int node_of_element_to_receive_from =
        db::get<Tags::NodeOfElementToReceiveFrom>(box);

    const auto& vector_0 = db::get<Tags::Vector0>(box);
    const auto& vector_1 = db::get<Tags::Vector1>(box);

    const auto& address_of_vector_0_on_sender =
        db::get<Tags::AddressOfVector0OnSender>(box);
    std::stringstream ss{};
    ss << vector_1.data();
    const std::string address_of_vector_1_on_receiver = ss.str();

    // Regardless of whether we are on different nodes or not, this should be
    // true
    SPECTRE_PARALLEL_REQUIRE(vector_0.size() ==
                             static_cast<size_t>(my_element) + 1);
    SPECTRE_PARALLEL_REQUIRE(vector_1.size() ==
                             static_cast<size_t>(element_to_receive_from) + 1);

    // If we are on the same node, we should have the same memory address for
    // our Vector1.data() and the senders Vector0.data(). But if we are on
    // different nodes, they will be different.
    if (node_of_element_to_receive_from == my_node) {
      SPECTRE_PARALLEL_REQUIRE(address_of_vector_0_on_sender ==
                               address_of_vector_1_on_receiver);
    } else {
      SPECTRE_PARALLEL_REQUIRE_FALSE(address_of_vector_0_on_sender ==
                                     address_of_vector_1_on_receiver);
    }

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <class Metavariables>
struct ArrayParallelComponent {
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<Initialize>>,
      Parallel::PhaseActions<Parallel::Phase::Solve,
                             tmpl::list<SendMessage, ReceiveMessage,
                                        Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<CheckMessage, Parallel::Actions::TerminatePhase>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  using array_index = int;

  static void allocate_array(
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<simple_tags_from_options>&
      /*initialization_items*/,
      const std::unordered_set<size_t>& /*procs_to_ignore*/ = {}) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    auto& array_proxy =
        Parallel::get_parallel_component<ArrayParallelComponent>(local_cache);

    if (sys::number_of_nodes() > 2) {
      ERROR(
          "The Test_AlgorithmMessages test must be run on one (1) or two (2) "
          "charm-nodes. For one node, you don't need any extra options. For 2 "
          "nodes, you'll need to add `+n2 +p2` to the submit command.");
    }

    for (int i = 0; i < total_number_of_array_elements; i++) {
      const int node = node_of(i, sys::number_of_nodes() == 2);
      array_proxy[i].insert(global_cache, {}, node);
    }
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ArrayParallelComponent>(local_cache)
        .start_phase(next_phase);
  }
};

struct TestMetavariables {
  using component_list = tmpl::list<ArrayParallelComponent<TestMetavariables>>;

  static constexpr const char* const help{
      "Test the receive_data entry method that uses Charm++ messages"};
  static constexpr bool ignore_unrecognized_command_line_options = false;

  static constexpr std::array<Parallel::Phase, 5> default_phase_order{
      {Parallel::Phase::Initialization, Parallel::Phase::Solve,
       Parallel::Phase::Testing, Parallel::Phase::Cleanup,
       Parallel::Phase::Exit}};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}
};
}  // namespace

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling, &setup_memory_allocation_failure_reporting,
    &disable_openblas_multithreading};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions, &enable_segfault_handler};

using charmxx_main_component = Parallel::Main<TestMetavariables>;

#include "Parallel/CharmMain.tpp"
