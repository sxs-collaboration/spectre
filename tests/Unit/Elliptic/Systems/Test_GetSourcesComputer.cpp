// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "Elliptic/Systems/GetSourcesComputer.hpp"

namespace elliptic {
namespace {
struct LinearSources {
  using argument_tags = tmpl::list<int>;
};
struct NonlinearSources {
  using argument_tags = tmpl::list<double>;
};

struct LinearSystem {
  using sources_computer = LinearSources;
};

struct NonlinearSystem {
  using sources_computer = NonlinearSources;
  using sources_computer_linearized = LinearSources;
};

struct NoSourcesSystem {
  using sources_computer = void;
  using sources_computer_linearized = void;
};

static_assert(
    std::is_same_v<get_sources_computer<LinearSystem, false>, LinearSources>);
static_assert(std::is_same_v<get_sources_argument_tags<LinearSystem, false>,
                             tmpl::list<int>>);
static_assert(
    std::is_same_v<get_sources_computer<LinearSystem, true>, LinearSources>);
static_assert(std::is_same_v<get_sources_argument_tags<LinearSystem, true>,
                             tmpl::list<int>>);
static_assert(std::is_same_v<get_sources_computer<NonlinearSystem, false>,
                             NonlinearSources>);
static_assert(std::is_same_v<get_sources_argument_tags<NonlinearSystem, false>,
                             tmpl::list<double>>);
static_assert(
    std::is_same_v<get_sources_computer<NonlinearSystem, true>, LinearSources>);
static_assert(std::is_same_v<get_sources_argument_tags<NoSourcesSystem, false>,
                             tmpl::list<>>);
static_assert(std::is_same_v<get_sources_argument_tags<NoSourcesSystem, true>,
                             tmpl::list<>>);
}  // namespace
}  // namespace elliptic
