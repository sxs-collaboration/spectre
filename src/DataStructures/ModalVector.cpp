// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/ModalVector.hpp"

#include <algorithm>                  // IWYU pragma: keep
#include <pup.h>                      // IWYU pragma: keep

#include "Utilities/StdHelpers.hpp"   // IWYU pragma: keep

/// Construct a ModalVector with value(s)
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VALUE(ModalVector)

/// \cond HIDDEN_SYMBOLS
/// Construct / Assign ModalVector with / to ModalVector reference or rvalue
// clang-tidy: calling a base constructor other than the copy constructor.
//             We reset the base class in reset_pointer_vector after calling its
//             default constructor
MAKE_EXPRESSION_VEC_DEF_CONSTRUCT_WITH_VEC(ModalVector)
/// \endcond

/// Define shift and (in)equivalence operators for ModalVector with itself
MAKE_EXPRESSION_VECMATH_OP_DEF_COMP_SELF(ModalVector)

/// Charm++ object packing / unpacking
MAKE_EXPRESSION_VEC_OP_PUP_CHARM(ModalVector)

/// \cond
template ModalVector::ModalVector(std::initializer_list<double> list);
/// \endcond
