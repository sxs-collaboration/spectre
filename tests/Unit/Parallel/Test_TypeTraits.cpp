// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "Parallel/Algorithms/AlgorithmArrayDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmGroupDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmNodegroupDeclarations.hpp"
#include "Parallel/Algorithms/AlgorithmSingletonDeclarations.hpp"
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
};

struct ArrayParallelComponent {
  using metavariables = Metavariables;
  using initialization_tags = tmpl::list<>;
};
struct GroupParallelComponent {
  using metavariables = Metavariables;
  using initialization_tags = tmpl::list<>;
};
struct NodegroupParallelComponent {
  using metavariables = Metavariables;
  using initialization_tags = tmpl::list<>;
};

using array_proxy = CProxy_AlgorithmArray<ArrayParallelComponent, int>;
using array_element_proxy = CProxyElement_AlgorithmArray<ArrayParallelComponent,
                                                         int>;
using group_proxy = CProxy_AlgorithmGroup<ArrayParallelComponent, int>;
using nodegroup_proxy = CProxy_AlgorithmNodegroup<ArrayParallelComponent, int>;
}  // namespace

static_assert(Parallel::is_array_proxy<array_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<array_element_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<group_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_array_proxy");

static_assert(not Parallel::is_array_element_proxy<array_proxy>::value,
              "Failed testing type trait is_array_element_proxy");
static_assert(Parallel::is_array_element_proxy<array_element_proxy>::value,
              "Failed testing type trait is_array_element_proxy");
static_assert(not Parallel::is_array_element_proxy<group_proxy>::value,
              "Failed testing type trait is_array_element_proxy");
static_assert(not Parallel::is_array_element_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_array_element_proxy");

static_assert(not Parallel::is_group_proxy<array_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<array_element_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(Parallel::is_group_proxy<group_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_group_proxy");

static_assert(not Parallel::is_node_group_proxy<array_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<array_element_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<group_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(Parallel::is_node_group_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_node_group_proxy");

// [has_pup_member_example]
static_assert(Parallel::has_pup_member<PupableClass>::value,
              "Failed testing type trait has_pup_member");
static_assert(Parallel::has_pup_member_t<PupableClass>::value,
              "Failed testing type trait has_pup_member");
static_assert(Parallel::has_pup_member_v<PupableClass>,
              "Failed testing type trait has_pup_member");
static_assert(not Parallel::has_pup_member<NonpupableClass>::value,
              "Failed testing type trait has_pup_member");
// [has_pup_member_example]

// [is_pupable_example]
static_assert(Parallel::is_pupable<PupableClass>::value,
              "Failed testing type trait is_pupable");
static_assert(Parallel::is_pupable_t<PupableClass>::value,
              "Failed testing type trait is_pupable");
static_assert(Parallel::is_pupable_v<PupableClass>,
              "Failed testing type trait is_pupable");
static_assert(not Parallel::is_pupable<NonpupableClass>::value,
              "Failed testing type trait is_pupable");
// [is_pupable_example]
