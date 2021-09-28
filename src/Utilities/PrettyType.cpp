// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/PrettyType.hpp"

#include <cstddef>
#include <string>
#include <string_view>

#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace pretty_type::detail {
namespace {
// Remove characters until a character in target is found.  This is
// used to skip characters we know we don't care about or do not
// understand.
void remove_until(const gsl::not_null<std::string_view*> remainder,
                  const char* targets) {
  const auto next_target = remainder->find_first_of(targets);
  ASSERT(next_target != remainder->npos,
         "Overran string " << *remainder << " looking for " << targets);
  remainder->remove_prefix(next_target);
}

std::string_view process_type_or_value(
    gsl::not_null<std::string_view*> remainder);
void process_literal(gsl::not_null<std::string_view*> remainder);
std::string_view process_item_sequence(
    gsl::not_null<std::string_view*> remainder);
void process_substitution(gsl::not_null<std::string_view*> remainder);
std::string_view process_identifier(gsl::not_null<std::string_view*> remainder);

// An entity of undetermined type.
std::string_view process_type_or_value(
    const gsl::not_null<std::string_view*> remainder) {
  remove_until(remainder, "0123456789LNS");
  std::string_view result = "";
  switch ((*remainder)[0]) {
    // Note that the first two cases return and the second two break.
    case 'L':
      process_literal(remainder);
      return "";
    case 'N':
      return process_item_sequence(remainder);
    case 'S':
      process_substitution(remainder);
      break;
    default:
      result = process_identifier(remainder);
      break;
  }
  // The last two cases can be templates, and so may be followed by
  // template parameters.
  if (not remainder->empty() and (*remainder)[0] == 'I') {
    process_item_sequence(remainder);
  }
  return result;
}

// A value (such as for a non-type template parameter) encoded as
// L<type><value>E or LDnE (nullptr)
void process_literal(const gsl::not_null<std::string_view*> remainder) {
  remainder->remove_prefix(1);
  if (not ('a' <= (*remainder)[0] and (*remainder)[0] <= 'z')) {
    if ((*remainder)[0] == 'D') {
      // decltype(something).  The only valid value of this form
      // should be nullptr.  The Itanium ABI standard says nullptr is
      // LDnE, but many compilers mangle it as LDn0E instead.
      ASSERT(
          (remainder->size() >= 3 and (*remainder)[1] == 'n' and
           (*remainder)[2] == 'E') or
              (remainder->size() >= 4 and (*remainder)[1] == 'n' and
               (*remainder)[2] == '0' and (*remainder)[3] == 'E'),
          "Expected nullptr constant (DnE or Dn0E) at start of " << *remainder);
    } else {
      // A named enum class
      process_type_or_value(remainder);
    }
  }
  remove_until(remainder, "E");
  remainder->remove_prefix(1);
}

// A sequence of items without separators, such as namespaces or
// template parameters.
std::string_view process_item_sequence(
    const gsl::not_null<std::string_view*> remainder) {
  remainder->remove_prefix(1);
  std::string_view result;
  for (;;) {
    // List from process_type_or_value with J and E added
    remove_until(remainder, "0123456789LNSJE");
    if ((*remainder)[0] == 'E') {
      // E = end
      remainder->remove_prefix(1);
      return result;
    } else if ((*remainder)[0] == 'J') {
      // A parameter pack.  Only allowed immediately inside a list of
      // template parameters, so doesn't have to be handled anywhere
      // else in the processing.
      process_item_sequence(remainder);
    } else {
      result = process_type_or_value(remainder);
    }
  }
  return result;
}

// A "substitution" (i.e., an abbreviation of some sort).  There are
// three types:
//   St          - "std::", which will be followed by more parts
//   S[a-z]      - some other standard library thing
//   S[0-9A-Z]*_ - a back-reference to some previously seen entity
// All cases where these could occur outside of template parameters
// were handled as special cases at the start, so we don't care about
// any of them.
void process_substitution(const gsl::not_null<std::string_view*> remainder) {
  remainder->remove_prefix(1);
  if ((*remainder)[0] == 't') {
    remainder->remove_prefix(1);
    process_type_or_value(remainder);
  } else if ('a' <= (*remainder)[0] and (*remainder)[0] <= 'z') {
    remainder->remove_prefix(1);
  } else {
    remove_until(remainder, "_");
    remainder->remove_prefix(1);
  }
}

// Identifier coded as <name length><name>
std::string_view process_identifier(
    const gsl::not_null<std::string_view*> remainder) {
  size_t length_chars = 0;
  const size_t identifier_chars =
      static_cast<size_t>(std::stoi(std::string(*remainder), &length_chars));
  std::string_view identifier =
      remainder->substr(length_chars, identifier_chars);
  remainder->remove_prefix(length_chars + identifier_chars);
  return identifier;
}
}  // namespace

std::string extract_short_name(const std::string& name) {
  // This ignores some of the less common language features (e.g., C
  // array types) and a lot of things that are not relevant to types.

  ASSERT(not name.empty(), "Cannot get type from an empty string");

  // Special-case a few standard library types.  These can be encoded
  // either using a special notation or using the standard namespace
  // notation, and some of them are type aliases in the latter case.
  // This makes them consistent.
  if (name == typeid(std::string).name()) {
    return "string";
  } else if (name == typeid(std::istream).name()) {
    return "istream";
  } else if (name == typeid(std::ostream).name()) {
    return "ostream";
  } else if (name == typeid(std::iostream).name()) {
    return "iostream";
  }

  // Short circuit for fundamentals and some standard library types.
  // These can occur in complicated types, but never as the component
  // we care about (which is always a struct or struct template name).
  if (name.size() == 1) {
    switch (name[0]) {
      case 'v': return "void";
      case 'b': return "bool";
      case 'c': return "char";
      case 'a': return "signed char";
      case 'h': return "unsigned char";
      case 's': return "short";
      case 't': return "unsigned short";
      case 'i': return "int";
      case 'j': return "unsigned int";
      case 'l': return "long";
      case 'm': return "unsigned long";
      case 'x': return "long long";
      case 'y': return "unsigned long long";
      case 'f': return "float";
      case 'd': return "double";
      case 'e': return "long double";
      default: ERROR("Builtin type " << name << " not handled");
    }
  } else if (name[0] == 'S') {
    // More possible standard library special cases, but with template
    // parameters.
    switch (name[1]) {
      case 'a': return "allocator";
      case 'b': return "basic_string";  // but not a std::string
      default:;
    }
  }

  std::string_view name_view(name);
  if (name[0] == 'S' and name[1] == 't') {
    // "St" is an abbreviation for the std:: prefix.  We don't care
    // about namespaces, so we can just drop it.
    name_view.remove_prefix(2);
  }

  std::string result(process_type_or_value(&name_view));
  ASSERT(name_view.empty(), "Type name was not fully processed");
  return result;
}
}  // namespace pretty_type::detail
