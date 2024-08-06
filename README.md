# nada-sandbox
Nada Sandbox is a browser-based interactive development environment (hosted using GitHub Pages at [`https://nillion.pub/nada-sandbox`](https://nillion.pub/nada-sandbox)) for authoring, sharing, and simulating [Nada programs](https://docs.nillion.com/nada-lang).

## Dependencies and Implementation

Nada Sandbox is implemented as a static webpage within the file `index.html`. No back-end server functionality is required for it to function, but users viewing the page must have internet access.

The interactive interface is implemented using pure JavaScript, while the interactive code editor is built using [CodeMirror](https://codemirror.net/). The Nada functionality relies on the [nada-dsl](https://pypi.org/project/nada-dsl/) library, which is loaded and executed directly within the browser using [PyScript](https://pyscript.net/).
