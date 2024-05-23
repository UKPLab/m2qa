# Installation
1. install nodejs dependencies: `npm install`
2. Optional: Since the labelstudio frontend npm package is outdated, we use a custom build. The code is located at: [static/src](static/src). In case you want to change this and create your own custom build:
    - Follow the instructions on the [labelstud.io website](https://labelstud.io/guide/frontend.html#Prepare-a-custom-LSF-build)
    - move the created `main.js` into `static/src/js/`
    - move the created `main.css` into `static/dist/css/`
3. Create Resources
    - create stylesheet: `npx tailwindcss -i ./static/src/css/style.css -o ./static/dist/css/style.css`
    - create javascript bundle: `npx webpack --config webpack.config.js`
