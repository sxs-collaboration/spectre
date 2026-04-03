// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines macros to allow serialization of abstract template base classes

#pragma once

#include <pup.h>

#ifdef __GNUC__
#pragma GCC system_header
#endif

#if defined(SPECTRE_USE_CHARM)
/*!
 * \ingroup ParallelGroup
 * \brief Mark derived classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * not a template class must contain this macro if it is to be serialized.
 */
#define WRAPPED_PUPable_decl_template(className)                \
  explicit className(CkMigrateMessage* /*unused*/) {}           \
  PUPable_decl_template(SINGLE_ARG(className))  // NOLINT

/*!
 * \ingroup ParallelGroup
 * \brief Mark derived template classes as serializable
 *
 * Any class that inherits from an abstract base class where the base class is
 * a template class must contain this macro if it is to be serialized.
 */
#define WRAPPED_PUPable_decl_base_template(baseClassName, className) \
  explicit className(CkMigrateMessage* /*unused*/) {}                \
  PUPable_decl_base_template(SINGLE_ARG(baseClassName), /* NOLINT */ \
                             SINGLE_ARG(className))     // NOLINT

/// Wraps the Charm++ macro, see the Charm++ documentation
#define WRAPPED_PUPable_abstract(className) \
  PUPable_abstract(SINGLE_ARG(className))  // NOLINT

/// \brief Used for the base class to inherit from.
#define SPECTRE_CHARM_PUPable(className) virtual PUP::able

/// \brief Used to inherit from the correct base classes when building with
/// Charm++
#define SPECTRE_CHARM_DERIVED(derivedName, baseName) baseName

/// Used to inherit from the correct base classes when building with findus when
/// the base class for serialization is several up in the hierarchy.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_DERIVED(derivedName, baseName)

/// \brief Only used for multi-inheritance auto-registration of derived classes
/// for serialization by findus. Empty in Charm++ mode.
///
/// \note findus supplies this macro directly.
#define FINDUS_OVERRIDE_SERIALIZATION_ID(Derived, Base)

/// \brief Expands to `virtual` if findus is used. Empty for Charm++.
#define SPECTRE_FINDUS_VIRTUAL()

/// \brief Expands to `override` if Charm++ is used. Empty for findus.
#define SPECTRE_CHARM_OVERRIDE() override

#elif defined(SPECTRE_USE_FINDUS)

/// \brief Used for the base class to inherit from.
#define SPECTRE_CHARM_PUPable(className) \
  virtual findus::serialize::SerializableBase<className>

/// Used to inherit from the correct base classes when building with Charm++
#define SPECTRE_CHARM_DERIVED(derivedName, baseName)                           \
  baseName, public virtual findus::serialize::SerializableDerived<derivedName, \
                                                                  baseName>

/// Used to inherit from the correct base classes when building with findus when
/// the base class for serialization is several up in the hierarchy.
///
/// Empty in Charm++ mode.
#define SPECTRE_FINDUS_DERIVED(derivedName, baseName) \
  , public virtual findus::serialize::SerializableDerived<derivedName, baseName>

/// \brief Interface with Charm++. Expands to empty.
#define WRAPPED_PUPable_decl_template(className) \
  static void unused_function_to_silence_compiler_warnings_for_charm_interop()

/// \brief Interface with Charm++. Expands to empty.
#define WRAPPED_PUPable_decl_base_template(baseClassName, className) \
  static void unused_function_to_silence_compiler_warnings_for_charm_interop()

/// \brief Interface with Charm++. Adds a virtual `pup` function to the base
/// class.
#define WRAPPED_PUPable_abstract(className) \
  virtual void pup(PUP::er& /*p*/) {}       \
  static void                               \
  unused_function_to_silence_compiler_warnings_for_charm_interop_abstract()

/// \brief Expands to `virtual` if findus is used. Empty for Charm++.
#define SPECTRE_FINDUS_VIRTUAL() virtual

/// \brief Expands to `override` if Charm++ is used. Empty for findus.
#define SPECTRE_CHARM_OVERRIDE()
#endif
