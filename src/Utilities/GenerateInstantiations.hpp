// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

// IWYU pragma: begin_exports
#include <boost/parameter/name.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/control/while.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/list/fold_left.hpp>
#include <boost/preprocessor/list/fold_right.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/list/size.hpp>
#include <boost/preprocessor/list/to_tuple.hpp>
#include <boost/preprocessor/list/transform.hpp>
#include <boost/preprocessor/logical/bitand.hpp>
#include <boost/preprocessor/logical/bool.hpp>
#include <boost/preprocessor/logical/compl.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/reverse.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <boost/preprocessor/variadic/elem.hpp>
#include <boost/preprocessor/variadic/to_list.hpp>
// IWYU pragma: end_exports

/// \cond
#define GENERATE_INSTANTIATIONS_DO_PRODUCT(INSTANTIATION_MACRO, LIST_OF_LISTS) \
  BOOST_PP_LIST_FOR_EACH_PRODUCT(INSTANTIATION_MACRO,                          \
                                 BOOST_PP_LIST_SIZE(LIST_OF_LISTS),            \
                                 BOOST_PP_LIST_TO_TUPLE(LIST_OF_LISTS))

#define GENERATE_INSTANTIATION_TUPLES_TO_LISTS(d, _, elem) \
  BOOST_PP_TUPLE_TO_LIST(BOOST_PP_TUPLE_SIZE(elem), elem)
/// \endcond

/*!
 * \ingroup UtilitiesGroup
 * \brief Macro useful for generating many explicit instantiations of function
 * or class templates
 *
 * It is often necessary to generate explicit instantiations of function or
 * class templates. Since the total number of explicit instantiations scales as
 * the product of the number of possible number of parameter values of each
 * template parameter, this quickly becomes tedious. This macro allows you to
 * easily generate hundreds of explicit instantiations.
 *
 * The first argument to the macro is a macro that takes two arguments and is
 * described below. The remaining arguments are macro-tuples, e.g. `(1, 2, 3)`.
 * The Cartesian product of the macro-tuples is then computed and each term is
 * passed as a tuple as the second argument to the `INSTANTIATION_MACRO`. The
 * first argument to the `INSTANTIATION_MACRO` is a Boost.Preprocessor internal
 * variable so just make it `_`. The `INSTANTIATION(_, data)` macro below serves
 * as an example. A concrete example is generating explicit instantiations of
 * the class `Index<Dim>` for `Dim = 0,1,2,3`, which you would do as follows:
 *
 * \code
 * #define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
 *
 * #define INSTANTIATION(_, data)         \
 *   template class Index<GET_DIM(data)>;
 *
 * GENERATE_INSTANTIATIONS(INSTANTIATION, (0, 1, 2, 3))
 *
 * #undef GET_DIM
 * #undef INSTANTIATION
 * \endcode
 *
 * This will generate:
 *
 * \code
 * template class Index<0>;
 * template class Index<1>;
 * template class Index<2>;
 * template class Index<3>;
 * \endcode
 *
 * It is also possible to generate explicit instantiations for multiple classes
 * or functions in a single call to `GENERATE_INSTANTIATIONS`. For example, the
 * (in)equivalence operators can be generated using:
 *
 * \code
 * #define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
 * #define GEN_OP(op, dim)                            \
 *   template bool operator op(const Index<dim>& lhs, \
 *                             const Index<dim>& rhs) noexcept;
 * #define INSTANTIATION(_, data)         \
 *   template class Index<GET_DIM(data)>; \
 *   GEN_OP(==, GET_DIM(data))            \
 *   GEN_OP(!=, GET_DIM(data))
 *
 * GENERATE_INSTANTIATIONS(INSTANTIATION, (0, 1, 2, 3))
 *
 * #undef GET_DIM
 * #undef GEN_OP
 * #undef INSTANTIATION
 * \endcode
 *
 * which will result in the instantiations:
 *
 * \code
 * template class Index<0>;
 * template bool operator==(const Index<0>& lhs, const Index<0>& rhs) noexcept;
 * template bool operator!=(const Index<0>& lhs, const Index<0>& rhs) noexcept;
 * template class Index<1>;
 * template bool operator==(const Index<1>& lhs, const Index<1>& rhs) noexcept;
 * template bool operator!=(const Index<1>& lhs, const Index<1>& rhs) noexcept;
 * template class Index<2>;
 * template bool operator==(const Index<2>& lhs, const Index<2>& rhs) noexcept;
 * template bool operator!=(const Index<2>& lhs, const Index<2>& rhs) noexcept;
 * template class Index<3>;
 * template bool operator==(const Index<3>& lhs, const Index<3>& rhs) noexcept;
 * template bool operator!=(const Index<3>& lhs, const Index<3>& rhs) noexcept;
 * \endcode
 *
 * Now let's look at generating instantiations of member function templates of
 * class templates, which will be a common use case. In this example we generate
 * explicit instantiations of all the member function templates of the class
 * `ScalarWave::Solutions::PlaneWave`. In total, for `Dim = 1,2,3` and types
 * `double` and `DataVector` this is about 42 explicit instantiations, which
 * would be extremely annoying to write by hand. The macro code is surprisingly
 * simple:
 *
 * \code
 * #define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
 * #define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
 *
 * #define INSTANTIATE(_, data)                                              \
 *   template Scalar<DTYPE(data)>                                            \
 *   ScalarWave::Solutions::PlaneWave<DIM(data)>::psi(                       \
 *       const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const noexcept; \
 *   template Scalar<DTYPE(data)>                                            \
 *   ScalarWave::Solutions::PlaneWave<DIM(data)>::dpsi_dt(                   \
 *       const tnsr::I<DTYPE(data), DIM(data)>& x, double t) const noexcept;
 *
 * GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector))
 *
 * #undef DIM
 * #undef DTYPE
 * #undef INSTANTIATE
 * \endcode
 *
 * We don't show the result from preprocessor since for all of the member
 * functions of `PlaneWave` the total output is approximately 150 lines, but you
 * can hopefully see the benefits of generating explicit instantiations using
 * the `GENERATE_INSTANTIATIONS` way.
 *
 * One thing that can be difficult is debugging metaprograms (be they template
 * or macro-based). To this end we provide a make target `DebugPreprocessor`
 * which prints the output of running the preprocessor on the file
 * `src/Executables/DebugPreprocessor/DebugPreprocessor.cpp`.
 * Note that the output of the `GENERATE_INSTANTIATIONS` macro will be on a
 * single line, so it often proves useful to copy-paste the output into an
 * editor and run clang-format over the code so it's easier to reason about.
 */
#define GENERATE_INSTANTIATIONS(INSTANTIATION_MACRO, ...)                \
  GENERATE_INSTANTIATIONS_DO_PRODUCT(                                    \
      INSTANTIATION_MACRO,                                               \
      BOOST_PP_LIST_TRANSFORM(GENERATE_INSTANTIATION_TUPLES_TO_LISTS, _, \
                              BOOST_PP_VARIADIC_TO_LIST(__VA_ARGS__)))
