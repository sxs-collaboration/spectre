# Distributed under the MIT License.
# See LICENSE.txt for details.

import itertools
import re
import sys

import gdb

if sys.version_info[0] > 2:
    Iterator = object
    imap = map
    izip = zip
    long = int
else:
    ### Python 2 stuff
    class Iterator:
        """Compatibility mixin for iterators

        Instead of writing next() methods for iterators, write
        __next__() methods and use this mixin to make them work in
        Python 2 as well as Python 3.

        Idea stolen from the "six" documentation:
        <http://pythonhosted.org/six/#six.Iterator>
        """

        def next(self):
            return self.__next__()

    from itertools import imap, izip


class VectorImplPrinter:
    """Print a VectorImpl, including DataVector, ComplexDataVector and their
    modal counterparts
    """

    class _iterator(Iterator):
        def __init__(self, start, num_entries):
            self.item = start
            self.finish = start + num_entries
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            count = self.count
            self.count = self.count + 1
            if self.item == self.finish:
                raise StopIteration
            elt = self.item.dereference()
            self.item = self.item + 1
            return ("[%d]" % count, elt)

    def __init__(self, val):
        self.val = val

    def children(self):
        return self._iterator(self.val["v_"], self.val["size_"])

    def to_string(self):
        return (
            str(self.val.type.tag)
            + " owning: "
            + str(self.val["owning_"])
            + " size: "
            + str(self.val["size_"])
            + " values "
        )

    def display_hint(self):
        return "array"


class GslSpanPrinter:
    "Print a gsl::span"

    class _iterator(Iterator):
        def __init__(self, start, num_entries):
            self.item = start
            self.finish = start + num_entries
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            count = self.count
            self.count = self.count + 1
            if self.item == self.finish:
                raise StopIteration
            elt = self.item.dereference()
            self.item = self.item + 1
            return ("[%d]" % count, elt)

    def __init__(self, val):
        self.val = val

    def children(self):
        return self._iterator(
            self.val["storage_"]["data_"], self.val["storage_"]["size_"]
        )

    def to_string(self):
        return (
            str(self.val.type.strip_typedefs().name).replace(", -1l", "")
            + " size: "
            + str(self.val["storage_"]["size_"])
            + " values "
        )

    def display_hint(self):
        return "array"


class TensorPrinter:
    "Print a Tensor"

    class _iterator(Iterator):
        def __init__(self, component_suffix_eval_string, start, num_entries):
            self.component_suffix_eval_string = component_suffix_eval_string
            self.item = start
            self.finish = start + int(num_entries)
            self.count = 0

        def __iter__(self):
            return self

        def __next__(self):
            count = self.count
            self.count = self.count + 1
            if self.item == self.finish:
                raise StopIteration
            elt = self.item.dereference()
            self.item = self.item + 1
            index = str(
                gdb.parse_and_eval(
                    self.component_suffix_eval_string + str(count) + ")"
                )
            )[2:-1]
            return ("\n[%s]" % index, elt)

    def children(self):
        array_name = str(
            self.val["data_"].type.strip_typedefs().fields()[0].name
        )
        return self._iterator(
            str(self.val.type.strip_typedefs()) + "::component_suffix(",
            self.val["data_"][array_name]
            .cast(self.val["data_"][array_name].type.strip_typedefs())[0]
            .address,
            int(
                self.val["data_"]
                .type.strip_typedefs()
                .fields()[0]
                .type.strip_typedefs()
                .range()[1]
            )
            + 1,
        )

    def __init__(self, val):
        if val.type.code == gdb.TYPE_CODE_TYPEDEF:
            self.typename = val.type.name
        else:
            self.typename = "Tensor"
        self.val = val

    def to_string(self):
        return self.typename


class VariablesPrinter:
    "Print a Variables"

    class _iterator(Iterator):
        def __init__(self, head, empty):
            self.head = head
            self.count = 0
            self.empty = empty

        def __iter__(self):
            return self

        def __next__(self):
            if self.empty or self.count == len(self.head.type.fields()):
                raise StopIteration
            count = self.count
            self.count = self.count + 1

            elet = self.head.cast(
                self.head.type.fields()[count].type.strip_typedefs()
            )["value_"]
            return (
                "[%s]"
                % str(self.head.type.fields()[count].type.strip_typedefs().name)
                .replace("tuples::tuples_detail::TaggedTupleLeaf<", "")
                .replace(", false>", "")
                .replace(", true>", ""),
                elet.cast(elet.type.strip_typedefs()),
            )

    def children(self):
        if self.is_empty:
            return self._iterator("", True)
        else:
            return self._iterator(
                self.val["reference_variable_data_"].cast(
                    self.val["reference_variable_data_"].type.strip_typedefs()
                ),
                False,
            )

    def __init__(self, val):
        self.typename = "Variables"
        self.val = val
        self.is_empty = False
        try:
            str(self.val["owning_"])
        except:
            self.is_empty = True

    def to_string(self):
        if self.is_empty:
            return "Variables<list<>>"
        else:
            return (
                "Variables: (owning:"
                + str(self.val["owning_"])
                + ",Tags:"
                + str(self.val["number_of_variables"])
                + ","
                + str(self.val["number_of_independent_components"])
                + "x"
                + str(self.val["number_of_grid_points_"])
                + ") "
            )


def spectre_build_pretty_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("spectre")
    pp.add_printer("DataVector", "^DataVector$", VectorImplPrinter)
    pp.add_printer(
        "ComplexDataVector", "^ComplexDataVector$", VectorImplPrinter
    )
    pp.add_printer("ModalVector", "^ModalVector$", VectorImplPrinter)
    pp.add_printer(
        "ComplexModalVector", "^ComplexModalVector$", VectorImplPrinter
    )
    pp.add_printer("gsl::span", "^gsl::span<.*-1.*>$", GslSpanPrinter)
    pp.add_printer("Scalar", "^Scalar<.*>$", TensorPrinter)
    pp.add_printer("tnsr", "^tnsr::.*>$", TensorPrinter)
    pp.add_printer("Tensor", "^Tensor<.*>$", TensorPrinter)
    pp.add_printer("Variables", "^Variables<.*>$", VariablesPrinter)
    return pp


if __name__ == "__main__":
    gdb.printing.register_pretty_printer(
        gdb.current_objfile(), spectre_build_pretty_printer()
    )
