// Distributed under the MIT License.
// See LICENSE.txt for details.

#ifndef SPECTRE_PCH_HPP
#define SPECTRE_PCH_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <codecvt>
#include <complex>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <exception>
#include <execinfo.h>
#include <filesystem>
#include <forward_list>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <iosfwd>
#include <istream>
#include <iterator>
#include <limits>
#include <list>
#include <locale>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <ostream>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <glob.h>
#include <libgen.h>
#include <unistd.h>

// Cannot include catch
// error: no member named 'Session' in namespace 'Catch'
// #include <catch.hpp>

#include <hdf5.h>

#include <libxsmm.h>

// Mac CI build couldn't find this
// #include <sharp_cxx.h>

#include <yaml-cpp/yaml.h>

#include <charm++.h>
#include <charm.h>
#include <ckarrayindex.h>
#include <converse.h>
#include <pup.h>
#include <pup_stl.h>

// Python bindings are not always built
// #include <pybind11/numpy.h>
// #include <pybind11/operators.h>
// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_fit.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix_double.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_poly.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_vector_double.h>

#include <brigand/adapted/fusion.hpp>
#include <brigand/adapted/variant.hpp>
#include <brigand/brigand.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/erase.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <boost/core/demangle.hpp>
#include <boost/functional/hash.hpp>
#include <boost/integer/common_factor_rt.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/math/interpolators/barycentric_rational.hpp>
#include <boost/math/quaternion.hpp>
#include <boost/math/special_functions/binomial.hpp>
#include <boost/math/special_functions/sign.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/tools/roots.hpp>
#include <boost/parameter/name.hpp>
#include <boost/preprocessor.hpp>
#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/arithmetic/sub.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/comparison/not_equal.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/control/iif.hpp>
#include <boost/preprocessor/control/while.hpp>
#include <boost/preprocessor/debug/assert.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/list/fold_left.hpp>
#include <boost/preprocessor/list/fold_right.hpp>
#include <boost/preprocessor/list/for_each.hpp>
#include <boost/preprocessor/list/for_each_product.hpp>
#include <boost/preprocessor/list/size.hpp>
#include <boost/preprocessor/list/to_tuple.hpp>
#include <boost/preprocessor/list/transform.hpp>
#include <boost/preprocessor/logical/bitand.hpp>
#include <boost/preprocessor/logical/bool.hpp>
#include <boost/preprocessor/logical/compl.hpp>
#include <boost/preprocessor/logical/not.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/punctuation/is_begin_parens.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/tuple/enum.hpp>
#include <boost/preprocessor/tuple/pop_front.hpp>
#include <boost/preprocessor/tuple/push_back.hpp>
#include <boost/preprocessor/tuple/push_front.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/tuple/reverse.hpp>
#include <boost/preprocessor/tuple/size.hpp>
#include <boost/preprocessor/tuple/to_array.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <boost/preprocessor/variadic/elem.hpp>
#include <boost/preprocessor/variadic/to_list.hpp>
#include <boost/preprocessor/variadic/to_tuple.hpp>
#include <boost/program_options.hpp>
#include <boost/range/combine.hpp>
#include <boost/rational.hpp>
// boost/stacktrace may not be used if libbacktrace is
// #include <boost/stacktrace.hpp>
// #include <boost/stacktrace/stacktrace_fwd.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/get.hpp>
#include <boost/vmd/is_empty.hpp>

#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/Column.h>
#include <blaze/math/CompressedMatrix.h>
#include <blaze/math/CompressedVector.h>
#include <blaze/math/CustomVector.h>
#include <blaze/math/DenseVector.h>
#include <blaze/math/DynamicMatrix.h>
#include <blaze/math/DynamicVector.h>
#include <blaze/math/GroupTag.h>
#include <blaze/math/Matrix.h>
#include <blaze/math/PaddingFlag.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/Submatrix.h>
#include <blaze/math/Subvector.h>
#include <blaze/math/TransposeFlag.h>
#include <blaze/math/Vector.h>
#include <blaze/math/constraints/SIMDPack.h>
#include <blaze/math/lapack/trsv.h>
#include <blaze/math/simd/BasicTypes.h>
#include <blaze/math/traits/MultTrait.h>
#include <blaze/math/typetraits/IsColumnMajorMatrix.h>
#include <blaze/math/typetraits/IsDenseMatrix.h>
#include <blaze/math/typetraits/IsSparseMatrix.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Vectorization.h>
#include <blaze/system/Version.h>
#include <blaze/util/typetraits/RemoveConst.h>

// The following includes must be in the following order, or we get
// error: no type named 'index' in namespace 'boost::detail::multi_array'
#include <boost/config.hpp>

#include <DataStructures/BoostMultiArray.hpp>
#include <boost/multi_array.hpp>
#include <boost/numeric/odeint.hpp>
// The preceeding includes...

#include <Utilities/GenerateInstantiations.hpp>
#include <Utilities/TMPL.hpp>

#endif  // SPECTRE_PCH_HPP
