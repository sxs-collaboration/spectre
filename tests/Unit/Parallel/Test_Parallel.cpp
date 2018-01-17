// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Parallel/Info.hpp"
#include "Parallel/Printf.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
struct TestStream {
  double a{1.0};
  std::vector<int> b{0, 4, 8, -7};
};

std::ostream& operator<<(std::ostream& os, const TestStream& t) {
  os << t.a << " (";
  for (size_t i = 0; i < t.b.size() - 1; ++i) {
    os << t.b[i] << ",";
  }
  os << t.b[t.b.size() - 1] << ")";
  return os;
}
}  // namespace

// [[OutputRegex, -100 3000000000 1.0000000000000000000e\+00 \(0,4,8,-7\) test 1
// 2 3 abf a o e u]]
SPECTRE_TEST_CASE("Unit.Parallel.printf", "[Unit][Parallel]") {
  const char c_string0[40] = {"test 1 2 3"};
  auto* c_string1 = new char[80];
  // clang-tidy: do not use pointer arithmetic
  c_string1[0] = 'a';   // NOLINT
  c_string1[1] = 'b';   // NOLINT
  c_string1[2] = 'f';   // NOLINT
  c_string1[3] = '\0';  // NOLINT
  constexpr const char* const c_string2 = {"a o e u"};
  Parallel::printf("%d %lld %s %s %s %s\n", -100, 3000000000, TestStream{},
                   c_string0, c_string1, c_string2);
  delete[] c_string1;
}

SPECTRE_TEST_CASE("Unit.Parallel.NodeAndPes", "[Unit][Parallel]") {
  CHECK(1 == Parallel::number_of_procs());
  CHECK(0 == Parallel::my_proc());
  CHECK(1 == Parallel::number_of_nodes());
  CHECK(0 == Parallel::my_node());
  CHECK(1 == Parallel::procs_on_node(Parallel::my_node()));
  CHECK(0 == Parallel::my_local_rank());
  CHECK(0 == Parallel::first_proc_on_node(Parallel::my_node()));
  CHECK(0 == Parallel::local_rank_of(Parallel::my_proc()));
  CHECK(0 == Parallel::node_of(Parallel::my_proc()));
  // We check that the wall time is greater than or equal to zero and less
  // than 2 seconds, just to check the function actually returns something.
  const double walltime = Parallel::wall_time();
  CHECK((0 <= walltime and 2 >= walltime));
}
