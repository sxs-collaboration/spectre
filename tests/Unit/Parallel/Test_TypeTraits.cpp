// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/TypeTraits.hpp"
#include "tests/Unit/Parallel/ParallelTestChares.hpp"

class PupableClass {
 public:
  void pup(PUP::er&) {}  // NOLINT
};
inline void operator|(PUP::er&, PupableClass&);  // NOLINT

class NonpupableClass {};

namespace {
struct MV {};
}  // namespace

static_assert(Parallel::is_array_proxy<CProxy_TestArrayChare<MV>>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<CProxy_TestChare<MV>>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<CProxy_TestGroupChare<MV>>::value,
              "Failed testing type trait is_array_proxy");
static_assert(
    not Parallel::is_array_proxy<CProxy_TestNodeGroupChare<MV>>::value,
    "Failed testing type trait is_array_proxy");

static_assert(not Parallel::is_chare_proxy<CProxy_TestArrayChare<MV>>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(Parallel::is_chare_proxy<CProxy_TestChare<MV>>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(not Parallel::is_chare_proxy<CProxy_TestGroupChare<MV>>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(
    not Parallel::is_chare_proxy<CProxy_TestNodeGroupChare<MV>>::value,
    "Failed testing type trait is_chare_proxy");

static_assert(not Parallel::is_group_proxy<CProxy_TestArrayChare<MV>>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<CProxy_TestChare<MV>>::value,
              "Failed testing type trait is_group_proxy");
static_assert(Parallel::is_group_proxy<CProxy_TestGroupChare<MV>>::value,
              "Failed testing type trait is_group_proxy");
static_assert(
    not Parallel::is_group_proxy<CProxy_TestNodeGroupChare<MV>>::value,
    "Failed testing type trait is_group_proxy");

static_assert(
    not Parallel::is_node_group_proxy<CProxy_TestArrayChare<MV>>::value,
    "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<CProxy_TestChare<MV>>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(
    not Parallel::is_node_group_proxy<CProxy_TestGroupChare<MV>>::value,
    "Failed testing type trait is_node_group_proxy");
static_assert(
    Parallel::is_node_group_proxy<CProxy_TestNodeGroupChare<MV>>::value,
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
