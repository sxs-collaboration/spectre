// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Protocols/Mutator.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace db::TestHelpers {

struct OptionTag {};

struct ExampleSimpleTag0 : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTag>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

struct ExampleSimpleTag1 : db::SimpleTag {
  using type = int;
  using option_tags = tmpl::list<OptionTag>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& option) { return option; }
};

struct ExampleSimpleTag2 : db::SimpleTag {
  using type = double;
};

// [mutator_protocol]
struct ExampleMutator : tt::ConformsTo<db::protocols::Mutator> {
  using return_tags = tmpl::list<ExampleSimpleTag0, ExampleSimpleTag1>;
  using argument_tags = tmpl::list<ExampleSimpleTag2>;

  static void apply(const gsl::not_null<double*> item_0,
                    const gsl::not_null<int*> item_1, const double item_2,
                    const int additional_arg) {
    *item_0 = 2.0 * item_2;
    *item_1 = 2 * additional_arg;
  }
};
// [mutator_protocol]
}  // namespace db::TestHelpers
