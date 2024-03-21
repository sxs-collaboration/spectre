// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Imex/Protocols/ImplicitSector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace imex::protocols {
/// Protocol for an IMEX evolution system.
///
/// In addition to the usual requirements for an evolution system, an
/// IMEX system must specify an `implicit_sectors` typelist of structs
/// conforming to protocols::ImplicitSector, each of which describes
/// an implicit solve to be performed during time steps.
///
/// For efficiency, the tags in the `tensors` type alias of each
/// sector are required to be adjacent in the system's variables.
///
/// \snippet DoImplicitStepSector.hpp ImexSystem
struct ImexSystem {
  template <typename ConformingType>
  struct test {
    using implicit_sectors = typename ConformingType::implicit_sectors;
    static_assert(
        tmpl::all<implicit_sectors,
                  tt::assert_conforms_to<tmpl::_1, ImplicitSector>>::value);

    template <typename Sector>
    struct sector_tensors {
      using type = typename Sector::tensors;
    };

    using variables_tag = typename ConformingType::variables_tag;
    static_assert(
        tmpl::all<
            implicit_sectors,
            std::is_same<
                tmpl::bind<tmpl::list_difference, sector_tensors<tmpl::_1>,
                           tmpl::pin<typename variables_tag::type::tags_list>>,
                tmpl::pin<tmpl::list<>>>>::value,
        "Implicit sector variables must be part of the system.");

    // There is a bug in brigand::lazy::reverse_find that it does not
    // actually evaluate its arguments lazily, so we must
    // bind<brigand::reverse_find> instead.
    static_assert(
        tmpl::all<
            implicit_sectors,
            std::is_same<
                tmpl::bind<
                    tmpl::reverse_find,
                    tmpl::lazy::find<
                        tmpl::pin<typename variables_tag::tags_list>,
                        tmpl::defer<std::is_same<
                            tmpl::_1,
                            tmpl::bind<tmpl::front, sector_tensors<tmpl::parent<
                                                        tmpl::_1>>>>>>,
                    tmpl::defer<std::is_same<
                        tmpl::_1,
                        tmpl::bind<tmpl::back,
                                   sector_tensors<tmpl::parent<tmpl::_1>>>>>>,
                sector_tensors<tmpl::_1>>>::value,
        "Tensors in an implicit sector must be adjacent in the system's "
        "variables.");
  };
};
}  // namespace imex::protocols
