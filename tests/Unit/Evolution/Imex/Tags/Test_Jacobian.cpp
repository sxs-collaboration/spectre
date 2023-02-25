// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Imex/Tags/Jacobian.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

class DataVector;

namespace {
struct Independent {
  using type = tnsr::i<DataVector, 2>;
};

struct Dependent {
  using type = tnsr::ii<DataVector, 2>;
};

struct Independent2;
struct Independent3;
struct Dependent2;
static_assert(
    std::is_same_v<
        imex::jacobian_tags<tmpl::list<Independent, Independent2, Independent3>,
                            tmpl::list<Dependent, Dependent2>>,
        tmpl::list<imex::Tags::Jacobian<Independent, Dependent>,
                   imex::Tags::Jacobian<Independent, Dependent2>,
                   imex::Tags::Jacobian<Independent2, Dependent>,
                   imex::Tags::Jacobian<Independent2, Dependent2>,
                   imex::Tags::Jacobian<Independent3, Dependent>,
                   imex::Tags::Jacobian<Independent3, Dependent2>>>);
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Imex.Tags.Jacobian", "[Unit][Evolution]") {
  using tag = imex::Tags::Jacobian<Independent, ::Tags::Source<Dependent>>;
  TestHelpers::db::test_simple_tag<tag>("Jacobian");
  static_assert(std::is_same_v<tag::type, tnsr::Ijj<DataVector, 2>>);
}
