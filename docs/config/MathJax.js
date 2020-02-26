window.MathJax = {
    tex: {
        tags: "ams",
        packages: ['base', 'ams', 'bbox', 'color', 'physics', 'newcommand']
    },
    options: {
        ignoreHtmlClass: 'tex2jax_ignore',
        processHtmlClass: 'tex2jax_process'
    },
    loader: {
        load: ['[tex]/bbox', '[tex]/color', '[tex]/physics', '[tex]/newcommand']
    }
};

(function () {
  var script = document.createElement('script');
  script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js';
  script.async = true;
  document.head.appendChild(script);
})();
