// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup_stl.h>

#include "Parallel/Info.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/System/Info.hpp"

namespace {
void test(const sys::Info& info) {
  CHECK(1 == info.number_of_procs());
  CHECK(0 == info.my_proc());
  CHECK(1 == info.number_of_nodes());
  CHECK(0 == info.my_node());
  CHECK(1 == info.procs_on_node(info.my_node()));
  CHECK(0 == info.my_local_rank());
  CHECK(0 == info.first_proc_on_node(info.my_node()));
  CHECK(0 == info.local_rank_of(info.my_proc()));
  CHECK(0 == info.node_of(info.my_proc()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Parallel.Info", "[Unit][Parallel]") {
  register_classes_with_charm<Parallel::Info>();
  const Parallel::Info info{};
  test(info);
  const auto pupped_info = serialize_and_deserialize(info);
  test(pupped_info);
  const auto cloned_info = info.get_clone();
  test(*cloned_info);
  const std::unique_ptr<Parallel::Info> info_ptr =
      std::make_unique<Parallel::Info>();
  test(*info_ptr);
  const auto pupped_info_ptr = serialize_and_deserialize(info_ptr);
  test(*pupped_info_ptr);
  const auto cloned_info_ptr = info_ptr->get_clone();
  test(*cloned_info_ptr);
  const std::unique_ptr<sys::Info> info_base_ptr =
      std::make_unique<Parallel::Info>();
  test(*info_base_ptr);
  const auto pupped_info_base_ptr = serialize_and_deserialize(info_base_ptr);
  test(*pupped_info_base_ptr);
  const auto cloned_info_base_ptr = info_base_ptr->get_clone();
  test(*cloned_info_base_ptr);
}
