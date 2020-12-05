// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macros to allow serialization of abstract template base classes

#pragma once

#include <pup.h>

#ifdef __GNUC__
#pragma GCC system_header
#endif

/*!
 * \ingroup ParallelGroup
 * \brief Mark derived classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * not a template class must contain this macro if it is to be serialized.
 */
#define WRAPPED_PUPable_decl_template(className) \
  PUPable_decl_template(SINGLE_ARG(className))  // NOLINT

/*!
 * \ingroup ParallelGroup
 * \brief Mark derived template classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * a template class must contain this macro if it is to be serialized.
 */
#define WRAPPED_PUPable_decl_base_template(baseClassName, className) \
  PUPable_decl_base_template(SINGLE_ARG(baseClassName), /* NOLINT */ \
                             SINGLE_ARG(className))     // NOLINT

/// Wraps the Charm++ macro, see the Charm++ documentation
#define WRAPPED_PUPable_decl(className) \
  PUPable_decl(SINGLE_ARG(className))  // NOLINT

/// Wraps the Charm++ macro, see the Charm++ documentation
#define WRAPPED_PUPable_abstract(className) \
  PUPable_abstract(SINGLE_ARG(className))  // NOLINT
