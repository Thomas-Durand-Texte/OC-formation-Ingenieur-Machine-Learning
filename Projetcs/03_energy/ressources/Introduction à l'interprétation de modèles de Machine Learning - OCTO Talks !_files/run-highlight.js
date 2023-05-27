document.addEventListener('DOMContentLoaded', (event) => {
  document.querySelectorAll('pre code').forEach((block) => {
    const languages = guessLanguages(block);
    hljs.highlightBlock(block, {
      languages: languages
    });
  });
  function guessLanguages(codeElement) {
    const prefix = /^lang(uage)?-/
    const preElement = codeElement.parentNode;
    const classes = classList(codeElement)
      .concat(classList(preElement))
      .filter((name) => prefix.test(name))
    if (!classes.length) {
      return;
    }
    return classes.map((name) => name.replace(prefix, ''));
  }
  function classList(element) {
    const list = [];
    for (var i=0; i<element.classList.length; ++i) {
      list.push(element.classList[i]);
    }
    return list;
  }
});

