// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/NumericInitialGuess.hpp"

namespace {

struct SomeNumericInitialGuess : elliptic::MarkAsNumericInitialGuess {};
struct NoNumericInitialGuess {};

static_assert(elliptic::is_numeric_initial_guess_v<SomeNumericInitialGuess>,
              "Failed testing elliptic::is_numeric_initial_guess_v");
static_assert(not elliptic::is_numeric_initial_guess_v<NoNumericInitialGuess>,
              "Failed testing elliptic::is_numeric_initial_guess_v");

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "FieldTag"; }
};

struct System {
  using fields_tag = ::Tags::Variables<tmpl::list<FieldTag>>;
};

static_assert(cpp17::is_same_v<
                  typename elliptic::NumericInitialGuess<System>::import_fields,
                  tmpl::list<FieldTag>>,
              "Failed testing elliptic::NumericInitialGuess");

}  // namespace
