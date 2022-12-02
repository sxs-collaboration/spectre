// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "Parallel/Algorithms/AlgorithmArray.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroup.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/TypeTraits.hpp"

namespace PUP {
class er;
}  // namespace PUP

namespace {
class PupableClass {
 public:
  void pup(PUP::er&) {}  // NOLINT
};

class NonpupableClass {};

struct Metavariables {
  enum class Phase { Initialization, Exit };
  using component_list = tmpl::list<>;
};

struct SingletonParallelComponent {
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Singleton;
};
struct ArrayParallelComponent {
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Array;
  using array_index = int;
};
struct GroupParallelComponent {
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Group;
};
struct NodegroupParallelComponent {
  using metavariables = Metavariables;
  using simple_tags_from_options = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Nodegroup;
};

using singleton_proxy =
    CProxy_AlgorithmSingleton<SingletonParallelComponent, int>;
using array_proxy = CProxy_AlgorithmArray<ArrayParallelComponent, int>;
using array_element_proxy =
    CProxyElement_AlgorithmArray<ArrayParallelComponent, int>;
using group_proxy = CProxy_AlgorithmGroup<GroupParallelComponent, int>;
using nodegroup_proxy =
    CProxy_AlgorithmNodegroup<NodegroupParallelComponent, int>;
}  // namespace

static_assert(Parallel::is_array_proxy<singleton_proxy>::value);
static_assert(Parallel::is_array_proxy<array_proxy>::value);
static_assert(not Parallel::is_array_proxy<array_element_proxy>::value);
static_assert(not Parallel::is_array_proxy<group_proxy>::value);
static_assert(not Parallel::is_array_proxy<nodegroup_proxy>::value);

static_assert(not Parallel::is_array_element_proxy<singleton_proxy>::value);
static_assert(not Parallel::is_array_element_proxy<array_proxy>::value);
static_assert(Parallel::is_array_element_proxy<array_element_proxy>::value);
static_assert(not Parallel::is_array_element_proxy<group_proxy>::value);
static_assert(not Parallel::is_array_element_proxy<nodegroup_proxy>::value);

static_assert(not Parallel::is_group_proxy<singleton_proxy>::value);
static_assert(not Parallel::is_group_proxy<array_proxy>::value);
static_assert(not Parallel::is_group_proxy<array_element_proxy>::value);
static_assert(Parallel::is_group_proxy<group_proxy>::value);
static_assert(not Parallel::is_group_proxy<nodegroup_proxy>::value);

static_assert(not Parallel::is_node_group_proxy<singleton_proxy>::value);
static_assert(not Parallel::is_node_group_proxy<array_proxy>::value);
static_assert(not Parallel::is_node_group_proxy<array_element_proxy>::value);
static_assert(not Parallel::is_node_group_proxy<group_proxy>::value);
static_assert(Parallel::is_node_group_proxy<nodegroup_proxy>::value);

// [has_pup_member_example]
static_assert(Parallel::has_pup_member<PupableClass>::value);
static_assert(Parallel::has_pup_member_t<PupableClass>::value);
static_assert(Parallel::has_pup_member_v<PupableClass>);
static_assert(not Parallel::has_pup_member<NonpupableClass>::value);
// [has_pup_member_example]

// [is_pupable_example]
static_assert(Parallel::is_pupable<PupableClass>::value);
static_assert(Parallel::is_pupable_t<PupableClass>::value);
static_assert(Parallel::is_pupable_v<PupableClass>);
static_assert(not Parallel::is_pupable<NonpupableClass>::value);
// [is_pupable_example]

static_assert(
    std::is_same_v<
        SingletonParallelComponent,
        Parallel::get_parallel_component_from_proxy<singleton_proxy>::type>);
static_assert(std::is_same_v<ArrayParallelComponent,
                             Parallel::get_parallel_component_from_proxy<
                                 array_element_proxy>::type>);
static_assert(std::is_same_v<
              GroupParallelComponent,
              Parallel::get_parallel_component_from_proxy<group_proxy>::type>);
static_assert(
    std::is_same_v<
        NodegroupParallelComponent,
        Parallel::get_parallel_component_from_proxy<nodegroup_proxy>::type>);

static_assert(Parallel::is_singleton_v<SingletonParallelComponent>);
static_assert(Parallel::is_array_v<ArrayParallelComponent>);
static_assert(Parallel::is_group_v<GroupParallelComponent>);
static_assert(Parallel::is_nodegroup_v<NodegroupParallelComponent>);
// These are special because they are (node)groups, but they don't run the
// Algorithm but they still have a `chare_type` type alias.
static_assert(
    Parallel::is_group_v<Parallel::MutableGlobalCache<Metavariables>>);
static_assert(Parallel::is_nodegroup_v<Parallel::GlobalCache<Metavariables>>);

static_assert(Parallel::is_singleton<SingletonParallelComponent>::value);
static_assert(Parallel::is_array<ArrayParallelComponent>::value);
static_assert(Parallel::is_group<GroupParallelComponent>::value);
static_assert(Parallel::is_nodegroup<NodegroupParallelComponent>::value);
static_assert(
    Parallel::is_group<Parallel::MutableGlobalCache<Metavariables>>::value);
static_assert(
    Parallel::is_nodegroup<Parallel::GlobalCache<Metavariables>>::value);
