\cond NEVER
Distributed under the MIT License.
See LICENSE.txt for details.
\endcond
# SFINAE {#sfinae}

SFINAE, Substitution Failure Is Not An Error, means that if a deduced template
substitution fails, compilation must continue. This can be exploited to make
decisions at compile time. See [here](http://nilsdeppe.com/posts/tmpl-part1)
for a discussion using `std::enable_if` to remove certain functions from
overload resolution or certain template specializations from name lookup.
Another method of controlling name lookup resolution is using `std::void_t`
which is implemented as `cpp17::void_t` in SpECTRE. `void_t` is a metafunction
from types to `void`, that is

```cpp
template <typename... Args>
using void_t = void;
```

`void_t` is useful when used in combination with `decltype` and `std::declval`
to probe if a type has certain members. For example, we can implement a type
trait to check if a type `T` is iterable by first have the general definition
inherit from `std::false_type` as follows,

```cpp
template <typename T, typename = void>
struct is_iterable : std::false_type {};
```

Next we will have specialization that uses `void_t` to check if the type `T`
has a `begin()` and `end()` function.

```cpp
template <typename T>
struct is_iterable<T, cpp17::void_t<decltype(std::declval<T>().begin(),
                                             std::declval<T>().end())>>
    : std::true_type {};
```

What is happening here? Well, we use `std::declval` to convert the type `T`
to a reference type, which allows us to call member functions inside `decltype`
expressions without having to construct an object. First we try to call the
member function `begin()` on `std::declval<T>()`, and if that succeeds we
throw away the result using the comma operator. Next we try to call `end()`
on `std::declval<T>()`, which, if it succeeds we get the return type of
using `decltype`. Note that `decltype` is important because we can only call
member functions on reference types inside of `decltype`, not just anywhere.
Finally, if all this succeeded use `void_t` to metareturn `void`, otherwise
the template parameters of `void_t` fail to evaluate and the specialization
cannot be resolved during name lookup. We could just as well use

```cpp
template <typename T>
struct is_iterable<T, cpp17::void_t<decltype(std::declval<T>().begin()),
                                    decltype(std::declval<T>().end())>>
    : std::true_type {};
```

Which of the two implementations of the `is_iterable` is preferred is simply
a matter of taste, both behave correctly.

If you're reading closely you might wonder why the `void_t` is necessary at
all, why not just `decltype(...)`? Well the reason is that since the default
template parameter metavalue is `void`, the specialization cannot be resolved
during name lookup unless the second template parameter in the specialization
is either `void` as well or is explicitly specified when the class template
is being invoked. Thus, the clearest implementation probably is

```cpp
template <typename T, typename = cpp17::void_t<>>
struct is_iterable : std::false_type {};

template <typename T>
struct is_iterable<T, cpp17::void_t<decltype(std::declval<T>().begin(),
                                             std::declval<T>().end())>>
    : std::true_type {};
```

You could now also define a helper type alias and constexpr boolean

```cpp
template <typename T>
using is_iterable_t = typename is_iterable<T>::type;

template <typename T>
constexpr bool is_iterable_v = is_iterable<T>::value;
```
