window.MathJax = {
    tex: {
        tags: "ams",
        packages: ['base', 'ams', 'bbox', 'color', 'physics', 'newcommand',
            'boldsymbol']
    },
    options: {
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
    },
    loader: {
        load: ['[tex]/bbox', '[tex]/color', '[tex]/physics', '[tex]/newcommand',
            '[tex]/boldsymbol']
    }
};

(function () {
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3.1.0/es5/tex-chtml-full.js';
  // cryptographic hashes can be found here:
  // https://www.jsdelivr.com/package/npm/mathjax?path=es5
  script.integrity = 'sha256-HJUiQvFxmEVWZ3D0qyz7Bg0JyJ2bkriI/WHQYo1ch5Y=';
  script.crossOrigin = 'anonymous';
  script.async = true;
  document.head.appendChild(script);
})();
