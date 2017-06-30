// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/Identity.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"

namespace EmbeddingMaps {
template class Identity<1>;
template class Identity<2>;
// Identity should only be used in ProductMaps if a particular dimension is
// unaffected.  So if the largest dim we do is 3, then you should never use
// Identity<3>
}  // namespace EmbeddingMaps
