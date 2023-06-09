// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes and functions for making classes creatable from
/// input files.

#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "Options/Context.hpp"
#include "Utilities/NoSuchType.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace YAML {
class Node;
}  // namespace YAML
/// \endcond

/// Utilities for parsing input files.
namespace Options {
/// \ingroup OptionParsingGroup
/// The type that options are passed around as.  Contains YAML node
/// data and an Context.
///
/// \note To use any methods on this class in a concrete function you
/// must include ParseOptions.hpp, but you do *not* need to include
/// that header to use this in an uninstantiated
/// `create_from_yaml::create` function.
class Option {
 public:
  const Context& context() const;

  /// Append a line to the contained context.
  void append_context(const std::string& context);

  /// Convert to an object of type `T`.
  template <typename T, typename Metavariables = NoSuchType>
  T parse_as() const;

  /// \note This constructor overwrites the mark data in the supplied
  /// context with the one from the node.
  ///
  /// \warning This method is for internal use of the option parser.
  explicit Option(YAML::Node node, Context context = {});

  /// \warning This method is for internal use of the option parser.
  explicit Option(Context context);

  /// \warning This method is for internal use of the option parser.
  const YAML::Node& node() const;

  /// Sets the node and updates the context's mark to correspond to it.
  ///
  /// \warning This method is for internal use of the option parser.
  void set_node(YAML::Node node);

 private:
  std::unique_ptr<YAML::Node> node_;
  Context context_;
};

/// \ingroup OptionParsingGroup
/// Used by the parser to create an object.  The default action is to
/// parse options using `T::options`.  This struct may be specialized
/// to change that behavior for specific types.
///
/// Do not call create directly.  Use Option::parse_as instead.
template <typename T>
struct create_from_yaml {
  template <typename Metavariables>
  static T create(const Option& options);
};

/// Provide multiple ways to construct a class.
///
/// This type may be included in an option list along with option
/// tags.  When creating the class, the parser will choose one of the
/// lists of options to use, depending on the user input.
///
/// The class must be prepared to accept any of the possible
/// alternatives as arguments to its constructor.  To disambiguate
/// multiple alternatives with the same types, a constructor may take
/// the full list of option tags expected as its first argument.
///
/// \snippet Test_Options.cpp alternatives
template <typename... AlternativeLists>
struct Alternatives {
  static_assert(sizeof...(AlternativeLists) >= 1,
                "Option alternatives must provide at least one alternative.");
  static_assert(
      tmpl::all<tmpl::list<tt::is_a<tmpl::list, AlternativeLists>...>>::value,
      "Option alternatives must be given as tmpl::lists.");
  static_assert(
      tmpl::none<
          tmpl::list<std::is_same<tmpl::list<>, AlternativeLists>...>>::value,
      "All option alternatives must have at least one option.");
};
}  // namespace Options
