// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/OptimizerHacks.hpp"

#if defined(__clang__) && __clang__ < 11
namespace optimizer_hacks {
void indicate_value_can_be_changed(void* /*variable*/) {}
}  // namespace optimizer_hacks
#endif  /* defined(__clang__) && __clang__ < 11 */

#if defined(__clang__) && __clang_major__ >= 15 && defined(__GLIBCXX__)
#include <string>
// clang v15+ fails to instantiate the following std library templates if
// pre-compiled headers are used
template std::__cxx11::basic_string<char, std::char_traits<char>,
                                    std::allocator<char>>::pointer
std::__cxx11::basic_string<char, std::char_traits<char>,
                           std::allocator<char>>::_M_use_local_data();
template std::__cxx11::basic_string<wchar_t, std::char_traits<wchar_t>,
                                    std::allocator<wchar_t>>::pointer
std::__cxx11::basic_string<wchar_t, std::char_traits<wchar_t>,
                           std::allocator<wchar_t>>::_M_use_local_data();
#endif /* defined(__clang__) && __clang_major__ >= 15 && defined(__GLIBCXX__) \
        */
