// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines classes for converting to and from Python objects

#pragma once

#include <Python.h>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <vector>
// Disable compiler warnings. NumPy ensures API compatibility among different
// 1.x versions, as features become deprecated in Numpy 1.x will still function
// but cause a compiler warning
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace pypp {
/// \cond
using None = void*;

template <typename T, typename = std::nullptr_t>
struct ToPyObject;

template <typename T>
PyObject* to_py_object(const T& t) {
  PyObject* value = ToPyObject<T>::convert(t);
  if (value == nullptr) {
    throw std::runtime_error{"Failed to convert argument."};
  }
  return value;
}

template <typename T, typename = std::nullptr_t>
struct FromPyObject;

template <typename T>
T from_py_object(PyObject* t) {
  return FromPyObject<T>::convert(t);
}
/// \endcond

/// Create a python tuple from Args
///
/// \tparam Args the types of the arguments to be put in the tuple (deducible)
/// \param t the arguments to put into the tuple
/// \return PyObject* containing a Python tuple
template <typename... Args>
PyObject* make_py_tuple(const Args&... t) {
  PyObject* py_tuple = PyTuple_New(sizeof...(Args));
  int entry = 0;
  const auto add_entry = [&entry, &py_tuple](const auto& arg) {
    PyObject* value = to_py_object(arg);
    PyTuple_SetItem(py_tuple, entry, value);
    entry++;
    return '0';
  };
  (void)add_entry; // GCC warns that add_entry is unused
  (void)std::initializer_list<char>{add_entry(t)...};
  return  py_tuple;
}

///\cond
template <>
struct ToPyObject<void, std::nullptr_t> {
  static PyObject* convert() { return Py_None; }
};

template <typename T>
struct ToPyObject<
    T, Requires<cpp17::is_same_v<typename std::decay<T>::type, std::string>>> {
  static PyObject* convert(const T& t) {
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    return PyString_FromString(t.c_str());
#elif PY_MAJOR_VERSION == 3
    return PyUnicode_FromString(t.c_str());
#else
    static_assert(false, "Only works on Python 2.7 and 3.x")
#endif
  }
};

template <typename T>
struct ToPyObject<
    T, Requires<cpp17::is_same_v<typename std::decay<T>::type, bool>>> {
  static PyObject* convert(const T& t) {
    return PyBool_FromLong(static_cast<long>(t));
  }
};

template <typename T>
struct ToPyObject<
    T, Requires<cpp17::is_same_v<typename std::decay<T>::type, int> or
                cpp17::is_same_v<typename std::decay<T>::type, short>>> {
  static PyObject* convert(const T& t) {
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    return PyInt_FromLong(t);
#elif PY_MAJOR_VERSION == 3
    return PyLong_FromLong(t);
#else
    static_assert(false, "Only works on Python 2.7 and 3.x")
#endif
  }
};

template <typename T>
struct ToPyObject<
    T, Requires<cpp17::is_same_v<typename std::decay<T>::type, long>>> {
  static PyObject* convert(const T& t) { return PyLong_FromLong(t); }
};

template <typename T>
struct ToPyObject<
    T, Requires<cpp17::is_same_v<typename std::decay<T>::type, unsigned long> or
                cpp17::is_same_v<typename std::decay<T>::type, unsigned int>>> {
  static PyObject* convert(const T& t) { return PyLong_FromUnsignedLong(t); }
};

template <typename T>
struct ToPyObject<
    T, Requires<
           cpp17::is_same_v<typename std::decay<T>::type, size_t> and
           not cpp17::is_same_v<typename std::decay<T>::type, unsigned long> and
           not cpp17::is_same_v<typename std::decay<T>::type, unsigned int>>> {
  static PyObject* convert(const T& t) { return PyLong_FromSize_t(t); }
};

template <typename T>
struct ToPyObject<
    T, Requires<std::is_floating_point<typename std::decay<T>::type>::value>> {
  static PyObject* convert(const T& t) { return PyFloat_FromDouble(t); }
};

template <typename T, typename A>
struct ToPyObject<std::vector<T, A>, std::nullptr_t> {
  static PyObject* convert(const std::vector<T, A>& t) {
    PyObject* list = PyList_New(static_cast<long>(t.size()));
    if (list == nullptr) {
      throw std::runtime_error{"Failed to convert argument."};
    }
    for (size_t i = 0; i < t.size(); ++i) {
      if (-1 ==
          PyList_SetItem(list, static_cast<long>(i), to_py_object<T>(t[i]))) {
        throw std::runtime_error{"Failed to add to PyList."};
      }
    }
    return list;
  }
};

template <typename T, size_t Size>
struct ToPyObject<std::array<T, Size>, std::nullptr_t> {
  static PyObject* convert(const std::array<T, Size>& t) {
    PyObject* list = PyList_New(static_cast<long>(t.size()));
    if (list == nullptr) {
      throw std::runtime_error{"Failed to convert argument."};
    }
    for (size_t i = 0; i < Size; ++i) {
      PyObject* value =
          ToPyObject<T>::convert(gsl::at(t, i));
      if (value == nullptr) {
        throw std::runtime_error{"Failed to convert argument."};
      }
      if (-1 == PyList_SetItem(list, static_cast<long>(i), value)) {
        throw std::runtime_error{"Failed to add to PyList."};
      }
    }
    return list;
  }
};

template <>
struct ToPyObject<DataVector, std::nullptr_t> {
  static PyObject* convert(const DataVector& t) {
    PyObject* npy_array = PyArray_SimpleNew(  // NOLINT
        1, (std::array<long, 1>{{static_cast<long>(t.size())}}.data()),
        NPY_DOUBLE);

    if (npy_array == nullptr) {
      throw std::runtime_error{"Failed to convert argument."};
    }
    for (size_t i = 0; i < t.size(); ++i) {
      // clang-tidy: Do not use pointer arithmetic
      // clang-tidy: Do not use reinterpret cast
      const auto data = static_cast<double*>(PyArray_GETPTR1(  // NOLINT
          reinterpret_cast<PyArrayObject*>(npy_array),         // NOLINT
          static_cast<long>(i)));
      if (data == nullptr) {
        throw std::runtime_error{"Failed to access argument of PyArray."};
      }
      *data = t[i];
    }
    return npy_array;
  }
};
struct FromPyObject<long, std::nullptr_t> {
  static long convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    } else if (not PyInt_Check(t) and not PyLong_Check(t)) {
#elif PY_MAJOR_VERSION == 3
    } else if (not PyLong_Check(t)) {
#else
    } else {
      static_assert(false, "Only works on Python 2.7 and 3.x")
#endif
      throw std::runtime_error{"Cannot convert non-long/int type to long."};
    }
    return PyLong_AsLong(t);
  }
};

template <>
struct FromPyObject<unsigned long, std::nullptr_t> {
  static unsigned long convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    } else if (not PyInt_Check(t) and not PyLong_Check(t)) {
#elif PY_MAJOR_VERSION == 3
    } else if (not PyLong_Check(t)) {
#else
    } else {
      static_assert(false, "Only works on Python 2.7 and 3.x");
#endif
      throw std::runtime_error{"Cannot convert non-long/int type to long."};
    }
    return PyLong_AsUnsignedLong(t);
  }
};

template <>
struct FromPyObject<double, std::nullptr_t> {
  static double convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    } else if (not PyFloat_Check(t)) {
      throw std::runtime_error{"Cannot convert non-double type to double."};
    }
    return PyFloat_AsDouble(t);
  }
};

template <>
struct FromPyObject<bool, std::nullptr_t> {
  static bool convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    } else if (not PyBool_Check(t)) {
      throw std::runtime_error{"Cannot convert non-bool type to bool."};
    }
    return static_cast<bool>(PyLong_AsLong(t));
  }
};

template <>
struct FromPyObject<std::string, std::nullptr_t> {
  static std::string convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    } else if (not PyString_CheckExact(t)) {
#elif PY_MAJOR_VERSION == 3
    } else if (not PyUnicode_CheckExact(t)) {
#else
    } else {
      static_assert(false, "Only works on Python 2.7 and 3.x")
#endif
      throw std::runtime_error{"Cannot convert non-string type to string."};
    }
#if PY_MAJOR_VERSION == 2 && PY_MINOR_VERSION == 7
    return std::string(PyString_AsString(t));
#elif PY_MAJOR_VERSION == 3
    PyObject* tascii = PyUnicode_AsASCIIString(t);
    if (nullptr == tascii) {
      throw std::runtime_error{"Cannot convert to ASCII string."};
    }
    std::string str = PyBytes_AsString(tascii);
    Py_DECREF(tascii);  // NOLINT
    return str;
#else
    static_assert(false, "Only works on Python 2.7 and 3.x")
#endif
  }
};

// This overload handles the case of converting from a python type of None to a
// void*
template <>
struct FromPyObject<void*, std::nullptr_t> {
  static void* convert(PyObject* t) {
    if (t == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    } else if (t != Py_None) {
      throw std::runtime_error{"Cannot convert non-None type to void."};
    }
    return nullptr;
  }
};

template <typename T>
struct FromPyObject<T, Requires<tt::is_a_v<std::vector, T>>> {
  static T convert(PyObject* p) {
    if (p == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    } else if (not PyList_CheckExact(p)) {
      throw std::runtime_error{"Cannot convert non-list type to vector."};
    }
    T t(static_cast<size_t>(PyList_Size(p)));
    for (size_t i = 0; i < t.size(); ++i) {
      PyObject* value = PyList_GetItem(p, static_cast<long>(i));
      if (value == nullptr) {
        throw std::runtime_error{"Failed to get argument from list."};
      }
      t[i] = from_py_object<typename T::value_type>(value);
    }
    return t;
  }
};

template <typename T>
struct FromPyObject<T, Requires<tt::is_std_array_v<T>>> {
  static T convert(PyObject* p) {
    if (p == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    } else if (not PyList_CheckExact(p)) {
      throw std::runtime_error{"Cannot convert non-list type to array."};
    }
    T t{};
    // clang-tidy: Do no implicitly decay an array into a pointer
    assert(PyList_Size(p) == static_cast<long>(t.size()));  // NOLINT
    for (size_t i = 0; i < t.size(); ++i) {
      PyObject* value = PyList_GetItem(p, static_cast<long>(i));
      if (value == nullptr) {
        throw std::runtime_error{"Failed to get argument from list."};
      }
      gsl::at(t, i) = from_py_object<typename T::value_type>(value);
    }
    return t;
  }
};
template <>
struct FromPyObject<DataVector, std::nullptr_t> {
  static DataVector convert(PyObject* p) {
    if (p == nullptr) {
      throw std::runtime_error{"Received null PyObject."};
    }
    // clang-tidy: c-style casts. (Expanded from macro)
    if (not PyArray_CheckExact(p)) {  // NOLINT
      throw std::runtime_error{"Cannot convert non-array type to DataVector."};
    }
    // clang-tidy: reinterpret_cast
    const auto npy_array = reinterpret_cast<PyArrayObject*>(p);  // NOLINT
    if (PyArray_TYPE(npy_array) != NPY_DOUBLE) {
      throw std::runtime_error{
          "Cannot convert array of non-double type to DataVector."};
    }
    if (PyArray_NDIM(npy_array) != 1) {
      throw std::runtime_error{
          "Cannot convert array of ndim != 1 to DataVector."};
    }
    // clang-tidy: c-style casts, pointer arithmetic. (Expanded from macro)
    DataVector t(static_cast<size_t>(PyArray_Size(p)));  // NOLINT
    for (size_t i = 0; i < t.size(); ++i) {
      // clang-tidy: pointer arithmetic. (Expanded from macro)
      const auto value = static_cast<const double*>(
          PyArray_GETPTR1(npy_array, static_cast<long>(i)));  // NOLINT
      if (value == nullptr) {
        throw std::runtime_error{"Failed to get argument from PyArray."};
      }
      t[i] = *value;
    }
    return t;
  }
};
///\endcond
}  // namespace pypp
