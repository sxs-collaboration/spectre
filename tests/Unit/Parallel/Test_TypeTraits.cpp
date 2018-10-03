// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <type_traits>

#include "AlgorithmArray.hpp"
#include "AlgorithmGroup.hpp"
#include "AlgorithmNodegroup.hpp"
#include "AlgorithmSingleton.hpp"
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

struct MV {};

struct SingletonParallelComponent {
  using metavariables = MV;
};
struct ArrayParallelComponent {
  using metavariables = MV;
};
struct GroupParallelComponent {
  using metavariables = MV;
};
struct NodegroupParallelComponent {
  using metavariables = MV;
};

using singleton_proxy =
    CProxy_AlgorithmSingleton<SingletonParallelComponent, int>;
using array_proxy = CProxy_AlgorithmArray<ArrayParallelComponent, int>;
using group_proxy = CProxy_AlgorithmGroup<ArrayParallelComponent, int>;
using nodegroup_proxy = CProxy_AlgorithmNodegroup<ArrayParallelComponent, int>;
}  // namespace

static_assert(Parallel::is_array_proxy<array_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<singleton_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<group_proxy>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_array_proxy");

static_assert(not Parallel::is_chare_proxy<array_proxy>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(Parallel::is_chare_proxy<singleton_proxy>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(not Parallel::is_chare_proxy<group_proxy>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(not Parallel::is_chare_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_chare_proxy");

static_assert(not Parallel::is_group_proxy<array_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<singleton_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(Parallel::is_group_proxy<group_proxy>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_group_proxy");

static_assert(not Parallel::is_node_group_proxy<array_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<singleton_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<group_proxy>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(Parallel::is_node_group_proxy<nodegroup_proxy>::value,
              "Failed testing type trait is_node_group_proxy");

/// [has_pup_member_example]
static_assert(Parallel::has_pup_member<PupableClass>::value,
              "Failed testing type trait has_pup_member");
static_assert(Parallel::has_pup_member_t<PupableClass>::value,
              "Failed testing type trait has_pup_member");
static_assert(Parallel::has_pup_member_v<PupableClass>,
              "Failed testing type trait has_pup_member");
static_assert(not Parallel::has_pup_member<NonpupableClass>::value,
              "Failed testing type trait has_pup_member");
/// [has_pup_member_example]

/// [is_pupable_example]
static_assert(Parallel::is_pupable<PupableClass>::value,
              "Failed testing type trait is_pupable");
static_assert(Parallel::is_pupable_t<PupableClass>::value,
              "Failed testing type trait is_pupable");
static_assert(Parallel::is_pupable_v<PupableClass>,
              "Failed testing type trait is_pupable");
static_assert(not Parallel::is_pupable<NonpupableClass>::value,
              "Failed testing type trait is_pupable");
/// [is_pupable_example]
