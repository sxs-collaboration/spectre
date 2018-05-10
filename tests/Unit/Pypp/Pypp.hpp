// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function pypp::call<R,Args...>

#pragma once

#include <Python.h>
#include <boost/range/combine.hpp>
#include <stdexcept>
#include <string>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Pypp/PyppFundamentals.hpp"

/// \ingroup TestingFrameworkGroup
/// Contains all functions for calling python from C++
namespace pypp {
namespace detail {

template <typename R, typename = std::nullptr_t>
struct CallImpl {
  template <typename... Args>
  static R call(const std::string& module_name,
                const std::string& function_name, const Args&... t) {
    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
                               module_name};
    }
    PyObject* func = PyObject_GetAttrString(module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }
    PyObject* args = pypp::make_py_tuple(t...);
    PyObject* value = PyObject_CallObject(func, args);
    Py_DECREF(args);  // NOLINT
    if (value == nullptr) {
      Py_DECREF(func);    // NOLINT
      Py_DECREF(module);  // NOLINT
      PyErr_Print();
      throw std::runtime_error{"Function returned null"};
    }
    auto ret = from_py_object<R>(value);
    Py_DECREF(value);   // NOLINT
    Py_DECREF(func);    // NOLINT
    Py_DECREF(module);  // NOLINT
    return ret;
  }
};

template <typename T>
struct convert_to_container_of_doubles {};

template <template <class, class...> class Container, class... Ts>
struct convert_to_container_of_doubles<Container<DataVector, Ts...>> {
  using type = Container<double, Ts...>;
};

template <size_t Dim>
struct convert_to_container_of_doubles<std::array<DataVector, Dim>> {
  using type = std::array<double, Dim>;
};

template <typename T>
using convert_to_container_of_doubles_t =
    typename convert_to_container_of_doubles<T>::type;

template <typename T>
struct SliceContainerImpl {
  static auto apply(const T& container_dv, const size_t slice_idx) noexcept {
    convert_to_container_of_doubles_t<std::decay_t<decltype(container_dv)>>
        container_double{};
    ASSERT(slice_idx < container_dv.begin()->size(),
           "Trying to slice DataVector of size " << container_dv.begin()->size()
                                                 << "with slice_idx "
                                                 << slice_idx);
    for (decltype(auto) double_and_datavector_components :
         boost::combine(container_double, container_dv)) {
      boost::get<0>(double_and_datavector_components) =
          boost::get<1>(double_and_datavector_components)[slice_idx];
    }
    return container_double;
  }
};

template <>
struct SliceContainerImpl<double> {
  static double apply(const double& t, const size_t /*slice_index*/) noexcept {
    return t;
  }
};

template <typename T>
decltype(auto) slice_container_of_datavectors_to_container_of_doubles(
    const T& in, const size_t slice_index) noexcept {
  return SliceContainerImpl<T>::apply(in, slice_index);
}

template <typename R>
struct CallImpl<
    R, Requires<(tt::is_a_v<Tensor, R> or tt::is_std_array_v<R>) and
                 cpp17::is_same_v<typename R::value_type, DataVector>>> {
  template <typename... Args>
  static R call(const std::string& module_name,
                const std::string& function_name, const Args&... t) {

    static_assert(sizeof...(Args) > 0,
                  "Call to python which returns a Tensor of DataVectors must "
                  "pass at least one argument");

    static_assert(
        cpp17::is_same_v<typename std::decay_t<decltype(
                             get_first_argument(t...))>::value_type,
                         DataVector>,
        "Call to python which returns a Tensor of DataVectors must "
        "pass a Tensor or std::array of DataVectors as its first argument.");

    PyObject* module = PyImport_ImportModule(module_name.c_str());
    if (module == nullptr) {
      PyErr_Print();
      throw std::runtime_error{std::string("Could not find python module.\n") +
                               module_name};
    }
    PyObject* func = PyObject_GetAttrString(module, function_name.c_str());
    if (func == nullptr or not PyCallable_Check(func)) {
      if (PyErr_Occurred()) {
        PyErr_Print();
      }
      throw std::runtime_error{"Could not find python function in module.\n"};
    }

    const auto put_container_of_doubles_into_container_of_datavector = [](
        auto& container_dv, const auto& container_double,
        const size_t slice_idx) noexcept {
      ASSERT(slice_idx < container_dv.begin()->size(),
             "Trying to slice DataVector of size "
                 << container_dv.begin()->size() << "with slice_idx "
                 << slice_idx);
      for (decltype(auto) datavector_and_double_components :
           boost::combine(container_dv, container_double)) {
        boost::get<0>(datavector_and_double_components)[slice_idx] =
            boost::get<1>(datavector_and_double_components);
      }
    };

    const size_t npts = get_first_argument(t...).begin()->size();
    auto return_container = make_with_value<R>(
        DataVector{npts, 0.}, std::numeric_limits<double>::signaling_NaN());

    for (size_t s = 0; s < npts; ++s) {
      PyObject* args = pypp::make_py_tuple(
          slice_container_of_datavectors_to_container_of_doubles(t, s)...);
      PyObject* value = PyObject_CallObject(func, args);
      Py_DECREF(args);  // NOLINT
      if (value == nullptr) {
        Py_DECREF(func);    // NOLINT
        Py_DECREF(module);  // NOLINT
        PyErr_Print();
        throw std::runtime_error{"Function returned null"};
      }

      const auto ret =
          from_py_object<convert_to_container_of_doubles_t<R>>(value);
      Py_DECREF(value);  // NOLINT
      put_container_of_doubles_into_container_of_datavector(return_container,
                                                            ret, s);
    }
    Py_DECREF(func);    // NOLINT
    Py_DECREF(module);  // NOLINT
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
/// Custom classes can be converted between Python and C++ by overloading the
/// `pypp::ToPythonObject<T>` and `pypp::FromPythonObject<T>` structs for your
/// own types. This tells C++ how to deconstruct the Python object into
/// fundamental types and reconstruct the C++ object and vice-versa.
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
template <typename R, typename... Args>
R call(const std::string& module_name, const std::string& function_name,
       const Args&... t) {
  return detail::CallImpl<R>::call(module_name, function_name, t...);
}
}  // namespace pypp
