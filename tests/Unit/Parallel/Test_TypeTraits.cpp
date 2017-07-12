// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/TypeTraits.hpp"
#include "tests/Unit/RunTests.hpp"

static_assert(Parallel::is_array_proxy<CProxy_TestArrayChare>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<CProxy_TestChare>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<CProxy_TestGroupChare>::value,
              "Failed testing type trait is_array_proxy");
static_assert(not Parallel::is_array_proxy<CProxy_TestNodeGroupChare>::value,
              "Failed testing type trait is_array_proxy");

static_assert(not Parallel::is_chare_proxy<CProxy_TestArrayChare>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(Parallel::is_chare_proxy<CProxy_TestChare>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(not Parallel::is_chare_proxy<CProxy_TestGroupChare>::value,
              "Failed testing type trait is_chare_proxy");
static_assert(not Parallel::is_chare_proxy<CProxy_TestNodeGroupChare>::value,
              "Failed testing type trait is_chare_proxy");

static_assert(not Parallel::is_group_proxy<CProxy_TestArrayChare>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<CProxy_TestChare>::value,
              "Failed testing type trait is_group_proxy");
static_assert(Parallel::is_group_proxy<CProxy_TestGroupChare>::value,
              "Failed testing type trait is_group_proxy");
static_assert(not Parallel::is_group_proxy<CProxy_TestNodeGroupChare>::value,
              "Failed testing type trait is_group_proxy");

static_assert(not Parallel::is_node_group_proxy<CProxy_TestArrayChare>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<CProxy_TestChare>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(not Parallel::is_node_group_proxy<CProxy_TestGroupChare>::value,
              "Failed testing type trait is_node_group_proxy");
static_assert(Parallel::is_node_group_proxy<CProxy_TestNodeGroupChare>::value,
              "Failed testing type trait is_node_group_proxy");
