// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/MockInfo.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/System/Info.hpp"

namespace {
class Expected {
 public:
  explicit Expected(const std::vector<int>& procs_per_node)
      : number_of_nodes_(static_cast<int>(procs_per_node.size())),
        number_of_procs_(alg::accumulate(procs_per_node, 0)),
        procs_per_node_(procs_per_node),
        first_proc_on_nodes_([&procs_per_node]() {
          std::vector<int> offsets(procs_per_node.size(), 0);
          for (size_t i = 0; i < offsets.size() - 1; ++i) {
            offsets[i + 1] = offsets[i] + procs_per_node[i];
          }
          return offsets;
        }()) {}
  int number_of_nodes() const { return number_of_nodes_; }
  int number_of_procs() const { return number_of_procs_; }
  int procs_on_node(int node) const {
    return procs_per_node_.at(static_cast<size_t>(node));
  }
  int global_proc(int node, int local_proc) const {
    return first_proc_on_nodes_.at(static_cast<size_t>(node)) + local_proc;
  }
  int first_proc_on_node(int node) const {
    return first_proc_on_nodes_.at(static_cast<size_t>(node));
  }
  int node_of(int global_proc) const {
    for (size_t i = 0; std::cmp_less(i, number_of_nodes_ - 1); ++i) {
      if (global_proc < first_proc_on_nodes_[i + 1]) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
  int local_rank_of(int global_proc) const {
    return global_proc -
           first_proc_on_nodes_.at(static_cast<size_t>(global_proc));
  }

 private:
  int number_of_nodes_;
  int number_of_procs_;
  std::vector<int> procs_per_node_;
  std::vector<int> first_proc_on_nodes_;
};

void test_test_case(const sys::Info& info, const Expected& expected,
                    const int my_node, const int my_local_core) {
  CAPTURE(my_node);
  CAPTURE(my_local_core);
  const int my_global_core = info.my_proc();
  CAPTURE(my_global_core);
  CHECK(expected.number_of_procs() == info.number_of_procs());
  CHECK(expected.global_proc(my_node, my_local_core) == my_global_core);
  CHECK(expected.number_of_nodes() == info.number_of_nodes());
  CHECK(my_node == info.my_node());
  CHECK(expected.procs_on_node(my_node) == info.procs_on_node(info.my_node()));
  CHECK(my_local_core == info.my_local_rank());
  CHECK(expected.first_proc_on_node(my_node) ==
        info.first_proc_on_node(info.my_node()));
  CHECK(my_node == info.node_of(my_global_core));
  CHECK(my_local_core == info.local_rank_of(my_global_core));
}

void test(const std::vector<int>& procs_per_core) {
  CAPTURE(procs_per_core);
  const Expected expected(procs_per_core);
  const int number_of_nodes = expected.number_of_nodes();
  CAPTURE(number_of_nodes);
  for (int n = 0; n < number_of_nodes; ++n) {
    for (int l = 0; l < procs_per_core.at(static_cast<size_t>(n)); ++l) {
      const ActionTesting::MockInfo info{procs_per_core, n, l};
      test_test_case(info, expected, n, l);
      const auto pupped_info = serialize_and_deserialize(info);
      test_test_case(pupped_info, expected, n, l);
      const auto cloned_info = info.get_clone();
      test_test_case(*cloned_info, expected, n, l);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ActionTesting.MockInfo", "[Unit]") {
  test(std::vector{1});
  test(std::vector{4});
  test(std::vector{3, 2});
  test(std::vector{1, 1, 1});
  test(std::vector{5, 0, 2, 4});
}
