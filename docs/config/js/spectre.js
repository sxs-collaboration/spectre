// Distributed under the MIT License.
// See LICENSE.txt for details.

window.onload = function(){

    // Hide type alias RHS that refers to a metafunction result in a (*_)detail namespace
    $("body").children().
    find("td.memItemRight, td.memTemplItemRight, td.memname").each(function () {
        $(this).html( $(this).html().replace(/( =[ ]+)[typedef ]*typename [A-Za-z0-9]*[_]?detail[s]?::[A-Za-z0-9\-_\.&;<> /;:\"=,#]+::[A-Za-z0-9_]+/g,
            "$1implementation defined") );
    });

    // Hide type alias RHS that refers to a type in a (*_)detail namespace
    // Also remove ... = decltype(something(_detail)::)
    $("body").children().find("td.memItemRight, td.memTemplItemRight, td.memname").each(function () {
        $(this).html( $(this).html().replace(/( =[ ]+)[typedef ]*[A-Za-z0-9\(]*[_]?detail[s]?::[A-Za-z0-9\-_\.&;<> /;:\"=,#\(\)]+/g,
            "$1implementation defined") );
    });

    // Hide unnamed template parameters: , typename = void
    // Not applied to div.memitem because that causes problems with rendering
    // MathJAX
    $("body").children().find("td.memItemRight, td.memTemplItemRight").each(function () {
        $(this).html( $(this).html().replace(/(template[&lt; ]+.*)(,[ ]*typename[ ]*=[ ]*void[ ]*)+/g,
            "$1") );
    });
    $("body").children().find("td.memItemRight, td.memTemplItemRight, div.title").each(function () {
        $(this).html( $(this).html().replace(/(&lt;[A-Za-z0-9 ,]+.*)(,[ ]*typename[ ]*&gt;)+/g,
            "$1&gt;") );
    });

    // Hide SFINAE in template parameters
    $("body").children().find("td.memTemplParams, div.memtemplate").each(function () {
        $(this).html( $(this).html().replace(/(template[A-Za-z0-9&;,\.=\(\) _]+)(,[ ]+typename[ ]+=[ ]+typename[ ]+std::enable_if.*::type)+/g,
            "$1") );

    });

    $("body").children().find("td.memTemplParams, div.memtemplate").each(function () {
        $(this).html( $(this).html().replace(/(template[A-Za-z0-9&;,\.=\(\) _]+)(,[ ]+typename[ ]+std::enable_if.*)+&gt;/g,
            "$1&gt;") );

    });

    // Hide enable_if_t for SFINAE
    $("body").children().find("td.memTemplParams, div.memtemplate").each(function () {
        $(this).html( $(this).html()
            .replace(/(template[A-Za-z0-9&;,\.=\(\) _]+)(,[ ]+std::enable_if_t.*)+&gt;/g,
                "$1&gt;") );

    });

    // Hide metafunctions that use only the template metaprogramming libraries
    $("body").children().find("td.memItemRight, td.memTemplItemRight").each(function () {
        $(this).html( $(this).html().replace(/( =[ ]+)tmpl::.*/g,
            "$1implementation defined") );
    });

    // Italicize "implementation defined"
    // Not applied to div.memitem because that causes problems with rendering
    // MathJAX
    $("body").children().find("td.memItemRight, td.memTemplItemRight").each(function () {
        $(this).html( $(this).html().replace(/implementation defined/g,
            "implementation defined".italics()) );
    });
};
