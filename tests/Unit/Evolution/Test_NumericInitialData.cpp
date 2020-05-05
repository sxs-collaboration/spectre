// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/NumericInitialData.hpp"
#include "Evolution/Protocols.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct SecondFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct System {
  using variables_tag = ::Tags::Variables<tmpl::list<FieldTag, SecondFieldTag>>;
};

// Test that the system's numeric initial data conforms to the protocol
static_assert(tt::assert_conforms_to<evolution::NumericInitialData<System>,
                                     evolution::protocols::NumericInitialData>);

// Test that the `import_fields` extracted from the system are correct
static_assert(std::is_same_v<
                  typename evolution::NumericInitialData<System>::import_fields,
                  tmpl::list<FieldTag, SecondFieldTag>>,
              "Failed testing evolution::NumericInitialData");

}  // namespace
