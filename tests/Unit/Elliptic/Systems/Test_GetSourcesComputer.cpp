// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "Elliptic/Systems/GetSourcesComputer.hpp"

namespace elliptic {
namespace {
struct LinearSources {};
struct NonlinearSources {};

struct LinearSystem {
  using sources_computer = LinearSources;
};

struct NonlinearSystem {
  using sources_computer = NonlinearSources;
  using sources_computer_linearized = LinearSources;
};

static_assert(
    std::is_same_v<get_sources_computer<LinearSystem, false>, LinearSources>);
static_assert(
    std::is_same_v<get_sources_computer<LinearSystem, true>, LinearSources>);
static_assert(std::is_same_v<get_sources_computer<NonlinearSystem, false>,
                             NonlinearSources>);
static_assert(
    std::is_same_v<get_sources_computer<NonlinearSystem, true>, LinearSources>);
}  // namespace
}  // namespace elliptic
