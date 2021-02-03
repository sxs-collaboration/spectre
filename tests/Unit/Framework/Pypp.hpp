// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function pypp::call<R,Args...>

#pragma once

#include <Python.h>
#include <array>
#include <boost/range/combine.hpp>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"

/*!
 * \ingroup TestingFrameworkGroup
 * \brief Contains all functions for calling python from C++
 */
namespace pypp {
namespace detail {
template <typename T, typename ConversionClass>
struct PackedTypeMatches
    : std::is_same<typename ConversionClass::packed_container, T> {};

template <typename T, typename ConversionClassList, typename = std::nullptr_t>
struct ContainerPackAndUnpack;

template <typename T, typename ConversionClassList>
struct ContainerPackAndUnpack<
    T, ConversionClassList,
    Requires<tmpl::found<ConversionClassList,
                         PackedTypeMatches<tmpl::pin<T>, tmpl::_1>>::value>>
    : tmpl::front<tmpl::find<ConversionClassList,
                             PackedTypeMatches<tmpl::pin<T>, tmpl::_1>>> {
  static_assert(
      tmpl::count_if<ConversionClassList,
                     PackedTypeMatches<tmpl::pin<T>, tmpl::_1>>::value == 1,
      "Found more than one conversion class for the type. See the first "
      "template parameter of ContainerPackAndUnpack for the type being "
      "converted, and the second template parameter for a list of all the "
      "additional conversion classes.");
};

template <typename ConversionClassList>
struct ContainerPackAndUnpack<DataVector, ConversionClassList, std::nullptr_t> {
  using unpacked_container = double;
  using packed_container = DataVector;
  using packed_type = packed_container;

  static inline unpacked_container unpack(
      const packed_container& packed, const size_t grid_point_index) noexcept {
    ASSERT(grid_point_index < packed.size(),
           "Trying to slice DataVector of size " << packed.size()
                                                 << " with grid_point_index "
                                                 << grid_point_index);
    return packed[grid_point_index];
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    (*packed)[grid_point_index] = unpacked;
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    return packed.size();
  }
};

template <typename ConversionClassList>
struct ContainerPackAndUnpack<ComplexDataVector, ConversionClassList,
                              std::nullptr_t> {
  using unpacked_container = std::complex<double>;
  using packed_container = ComplexDataVector;
  using packed_type = packed_container;

  static inline unpacked_container unpack(
      const packed_type& packed, const size_t grid_point_index) noexcept {
    ASSERT(grid_point_index < packed.size(),
           "Trying to slice ComplexDataVector of size "
               << packed.size() << " with grid_point_index "
               << grid_point_index);
    return packed[grid_point_index];
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    (*packed)[grid_point_index] = unpacked;
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    return packed.size();
  }
};

template <typename T, typename... Ts, typename ConversionClassList>
struct ContainerPackAndUnpack<Tensor<T, Ts...>, ConversionClassList,
                              std::nullptr_t> {
  using unpacked_container =
      Tensor<typename ContainerPackAndUnpack<
                 T, ConversionClassList>::unpacked_container,
             Ts...>;
  using packed_container = Tensor<
      typename ContainerPackAndUnpack<T, ConversionClassList>::packed_container,
      Ts...>;
  using packed_type =
      typename ContainerPackAndUnpack<T, ConversionClassList>::packed_type;

  static inline unpacked_container unpack(
      const packed_container& packed, const size_t grid_point_index) noexcept {
    unpacked_container unpacked{};
    for (size_t storage_index = 0; storage_index < unpacked.size();
         ++storage_index) {
      unpacked[storage_index] =
          ContainerPackAndUnpack<T, ConversionClassList>::unpack(
              packed[storage_index], grid_point_index);
    }
    return unpacked;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    for (size_t storage_index = 0; storage_index < unpacked.size();
         ++storage_index) {
      ContainerPackAndUnpack<T, ConversionClassList>::pack(
          make_not_null(&(*packed)[storage_index]), unpacked[storage_index],
          grid_point_index);
    }
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    return ContainerPackAndUnpack<T, ConversionClassList>::get_size(packed[0]);
  }
};

template <typename T, size_t Size, typename ConversionClassList>
struct ContainerPackAndUnpack<std::array<T, Size>, ConversionClassList,
                              std::nullptr_t> {
  using unpacked_container =
      std::array<typename ContainerPackAndUnpack<
                     T, ConversionClassList>::unpacked_container,
                 Size>;
  using packed_container = std::array<
      typename ContainerPackAndUnpack<T, ConversionClassList>::packed_container,
      Size>;
  using packed_type =
      typename ContainerPackAndUnpack<T, ConversionClassList>::packed_type;

  static inline unpacked_container unpack(
      const packed_container& packed, const size_t grid_point_index) noexcept {
    unpacked_container unpacked{};
    for (size_t i = 0; i < Size; ++i) {
      gsl::at(unpacked, i) =
          ContainerPackAndUnpack<T, ConversionClassList>::unpack(
              gsl::at(packed, i), grid_point_index);
    }
    return unpacked;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    for (size_t i = 0; i < unpacked.size(); ++i) {
      ContainerPackAndUnpack<T, ConversionClassList>::pack(
          make_not_null(&gsl::at(*packed, i)), gsl::at(unpacked, i),
          grid_point_index);
    }
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    return ContainerPackAndUnpack<T, ConversionClassList>::get_size(packed[0]);
  }
};

// scalars are the only sort of spin-weighted type we support.
template <typename ValueType, int Spin, typename ConversionClassList>
struct ContainerPackAndUnpack<Scalar<SpinWeighted<ValueType, Spin>>,
                              ConversionClassList, std::nullptr_t> {
  using unpacked_container = Scalar<typename ContainerPackAndUnpack<
      ValueType, ConversionClassList>::unpacked_container>;
  using packed_container =
      Scalar<SpinWeighted<typename ContainerPackAndUnpack<
                              ValueType, ConversionClassList>::packed_container,
                          Spin>>;
  using packed_type =
      typename ContainerPackAndUnpack<ValueType,
                                      ConversionClassList>::packed_type;

  static inline unpacked_container unpack(
      const packed_container& packed, const size_t grid_point_index) noexcept {
    unpacked_container unpacked{};
    get(unpacked) =
        ContainerPackAndUnpack<ValueType, ConversionClassList>::unpack(
            get(packed).data(), grid_point_index);
    return unpacked;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    ContainerPackAndUnpack<ValueType, ConversionClassList>::pack(
        make_not_null(&(get(*packed).data())), get(unpacked), grid_point_index);
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    return ContainerPackAndUnpack<ValueType, ConversionClassList>::get_size(
        packed[0].data());
  }
};

template <typename T, typename ConversionClassList>
struct ContainerPackAndUnpack<std::optional<T>, ConversionClassList,
                              std::nullptr_t> {
  using unpacked_container = std::optional<typename ContainerPackAndUnpack<
      T, ConversionClassList>::unpacked_container>;
  using packed_container = std::optional<typename ContainerPackAndUnpack<
      T, ConversionClassList>::packed_container>;
  using packed_type =
      typename ContainerPackAndUnpack<T, ConversionClassList>::packed_type;

  static inline unpacked_container unpack(
      const packed_container& packed, const size_t grid_point_index) noexcept {
    if (static_cast<bool>(packed)) {
      return unpacked_container{
          ContainerPackAndUnpack<T, ConversionClassList>::unpack(
              *packed, grid_point_index)};
    }
    return unpacked_container{std::nullopt};
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container& unpacked,
                          const size_t grid_point_index) {
    if ((std::is_same_v<packed_type, DataVector> or
         std::is_same_v<
             packed_type,
             ComplexDataVector>)and UNLIKELY(not static_cast<bool>(unpacked))) {
      throw std::runtime_error(
          "Returned type is None in one element of the std::optional's "
          "packed type (DataVector or ComplexDataVector). We can't support "
          "this because we can't just make one element of the packed type be "
          "an invalid optional.");
    }
    ContainerPackAndUnpack<T, ConversionClassList>::pack(
        make_not_null(&*packed), unpacked, grid_point_index);
  }

  static inline size_t get_size(const packed_container& packed) noexcept {
    if (static_cast<bool>(packed)) {
      return ContainerPackAndUnpack<T, ConversionClassList>::get_size(*packed);
    }
    return 1;
  }
};

template <typename T, typename ConversionClassList>
struct ContainerPackAndUnpack<
    T, ConversionClassList,
    Requires<std::is_floating_point_v<T> or std::is_integral_v<T> or
             std::is_same_v<T, std::string>>> {
  using unpacked_container = T;
  using packed_container = T;
  using packed_type = packed_container;

  static inline unpacked_container unpack(
      const packed_container t, const size_t /*grid_point_index*/) noexcept {
    return t;
  }

  static inline void pack(const gsl::not_null<packed_container*> packed,
                          const unpacked_container unpacked,
                          const size_t /*grid_point_index*/) {
    *packed = unpacked;
  }

  static inline size_t get_size(const packed_container& /*packed*/) noexcept {
    return 1;
  }
};

template <typename ReturnType>
ReturnType call_work(PyObject* python_module, PyObject* func, PyObject* args) {
  PyObject* value = PyObject_CallObject(func, args);
  Py_DECREF(args);  // NOLINT
  if (value == nullptr) {
    Py_DECREF(func);           // NOLINT
    Py_DECREF(python_module);  // NOLINT
    PyErr_Print();
    throw std::runtime_error{"Function returned null"};
  }

  auto ret = from_py_object<ReturnType>(value);
  Py_DECREF(value);  // NOLINT
  return ret;
}

template <typename ReturnType, typename ConversionClassList,
          typename = std::nullptr_t>
struct CallImpl {
  template <typename... Args>
  static ReturnType call(const std::string& module_name,
                         const std::string& function_name, const Args&... t) {
    PyObject* python_module = PyImport_ImportModule(module_name.c_str());
    if (python_module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
                               module_name};
    }
    PyObject* func =
        PyObject_GetAttrString(python_module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }
    PyObject* args = pypp::make_py_tuple(t...);
    auto ret = call_work<ReturnType>(python_module, func, args);
    Py_DECREF(func);           // NOLINT
    Py_DECREF(python_module);  // NOLINT
    return ret;
  }
};

template <typename ReturnType, typename ConversionClassList>
struct CallImpl<
    ReturnType, ConversionClassList,
    Requires<
        (tt::is_a_v<Tensor, ReturnType> or tt::is_std_array_v<ReturnType>)and(
            std::is_same_v<typename ContainerPackAndUnpack<
                               ReturnType, ConversionClassList>::packed_type,
                           DataVector> or
            std::is_same_v<typename ContainerPackAndUnpack<
                               ReturnType, ConversionClassList>::packed_type,
                           ComplexDataVector>)>> {
  template <typename... Args>
  static ReturnType call(const std::string& module_name,
                         const std::string& function_name, const Args&... t) {
    static_assert(sizeof...(Args) > 0,
                  "Call to python which returns a Tensor of DataVectors must "
                  "pass at least one argument");

    PyObject* python_module = PyImport_ImportModule(module_name.c_str());
    if (python_module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
                               module_name};
    }
    PyObject* func =
        PyObject_GetAttrString(python_module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }

    const std::array<size_t, sizeof...(Args)> arg_sizes{
        {ContainerPackAndUnpack<Args, ConversionClassList>::get_size(t)...}};
    const size_t npts = *std::max_element(arg_sizes.begin(), arg_sizes.end());
    for (size_t i = 0; i < arg_sizes.size(); ++i) {
      if (gsl::at(arg_sizes, i) != 1 and gsl::at(arg_sizes, i) != npts) {
        ERROR(
            "Each argument must return size 1 or "
            << npts
            << " (the number of points in the DataVector), but argument number "
            << i << " has size " << gsl::at(arg_sizes, i));
      }
    }
    auto return_container = make_with_value<typename ContainerPackAndUnpack<
        ReturnType, ConversionClassList>::packed_container>(npts, 0.0);

    for (size_t s = 0; s < npts; ++s) {
      PyObject* args = pypp::make_py_tuple(
          ContainerPackAndUnpack<Args, ConversionClassList>::unpack(t, s)...);
      auto ret = call_work<typename ContainerPackAndUnpack<
          ReturnType, ConversionClassList>::unpacked_container>(python_module,
                                                                func, args);
      ContainerPackAndUnpack<ReturnType, ConversionClassList>::pack(
          make_not_null(&return_container), ret, s);
    }
    Py_DECREF(func);           // NOLINT
    Py_DECREF(python_module);  // NOLINT
    return return_container;
  }
};
}  // namespace detail

/// Calls a Python function from a module/file with given parameters
///
/// \param module_name name of module the function is in
/// \param function_name name of Python function in module
/// \param t the arguments to be passed to the Python function
/// \return the object returned by the Python function converted to a C++ type
///
/// Custom conversion from containers to basic types that can be converted to
/// python objects via the `pypp::ToPyObject` class can be added by passing a
/// typelist of conversion class as the second template parameter to
/// `pypp::call`. This is generally used for converting classes like
/// `Tensor<DataVector, ...>` to a `Tensor<double, ...>` to support using
/// `numpy.einsum` in the python code. However, this can also be used to convert
/// a complicated class such as an equation of state to a bunch of numbers that
/// the python code for the particular test can use to compute the required data
/// without having to fully implement python bindings. The conversion classes
/// must have the following:
/// - a type alias `unpacked_container` that is the type at a single grid point.
/// - a type alias `packed_container` the type used when converting the result
///   from the python call back into the C++ code.
/// - a type alias `packed_type` which corresponds to the packed type that the
///   high-level container holds. For example, a `Scalar<DataVector>` would have
///   `packed_type` being `DataVector`, a `std::vector<Scalar<DataVector>>`
///   would have `packed_type` being `DataVector`, and a
///   `std::vector<Scalar<double>>` would have `packed_type` being `double`.
/// - an `unpack` function that takes as its arguments the `packed_container`
///   and the `grid_point_index`, and returns an `unpacked_container` object
/// - a `pack` function that takes as its arguments a
///   `gsl::not_null<packed_container*>`, an `const unpacked_container&`, and a
///   `size_t` corresponding to which grid point to pack
/// - a `get_size` function that takes as its only argument an object of
///   `packed_container` and returns the number of elements in the
///   `packed_type`
///
/// Examples of conversion classes can be found in the specializations of
/// `ContainerPackAndUnpack` in the `Pypp.hpp` file. Below is an example of a
/// conversion class that takes a class (`ClassForConversionTest`) and returns
/// the `a_` member variable.
///
/// \snippet Test_Pypp.cpp convert_arbitrary_a
///
/// Here is the call to `pypp::call`:
///
/// \snippet Test_Pypp.cpp convert_arbitrary_a_call
///
/// A conversion class for retrieving the member variables `b_` can also be
/// written as follows:
///
/// \snippet Test_Pypp.cpp convert_arbitrary_b
///
/// \note In order to setup the python interpreter and add the local directories
/// to the path, a SetupLocalPythonEnvironment object needs to be constructed
/// in the local scope.
///
/// \example
/// The following example calls the function `test_numeric` from the module
/// `pypp_py_tests` which multiplies two integers.
/// \snippet Test_Pypp.cpp pypp_int_test
/// Alternatively, this examples calls `test_vector` from `pypp_py_tests` which
/// converts two vectors to python lists and multiplies them pointwise.
/// \snippet Test_Pypp.cpp pypp_vector_test
///
/// Pypp can also be used to take a function that performs manipulations of
/// NumPy arrays and apply it to either a Tensor of doubles or a Tensor of
/// DataVectors. This is useful for testing functions which act on Tensors
/// pointwise. For example, let's say we wanted to call the NumPy function which
/// performs \f$ v_i = A B^a C_{ia} + D^{ab} E_{iab} \f$, which is implemented
/// in python as
///
/// \code{.py} def test_einsum(scalar, t_A, t_ia, t_AA, t_iaa):
///    return scalar * np.einsum("a,ia->i", t_A, t_ia) +
///           np.einsum("ab, iab->i", t_AA, t_iaa)
/// \endcode
///
/// where \f$ v_i \f$ is the return tensor and
/// \f$ A, B^a, C_{ia},D^{ab}, E_{iab} \f$ are the input tensors respectively.
/// We call this function through C++ as:
/// \snippet Test_Pypp.cpp einsum_example
/// for type `T` either a `double` or `DataVector`.
///
/// Pypp will also support testing of functions which return and operate on
/// `std::array`s of `DataVectors`s. To return a `std::array` of DataVectors,
/// the python function should return a python list of doubles.
///
/// \note In order to return a
/// `Tensor<DataVector...>` from `pypp::call`, at least one
/// `Tensor<DataVector...>` must be taken as an argument, as the size of the
/// returned tensor needs to be deduced.
template <typename ReturnType, typename ConversionClassList = tmpl::list<>,
          typename... Args>
ReturnType call(const std::string& module_name,
                const std::string& function_name, const Args&... t) {
  return detail::CallImpl<ReturnType, ConversionClassList>::call(
      module_name, function_name, t...);
}
}  // namespace pypp
