const path = require('path');

module.exports = {
  "mode": "none",
  "entry": "./static/src/js/index.js",
  resolve: {
    modules: [path.resolve(__dirname, './node_modules')]
  },
  devtool: 'source-map',
  "output": {
    "path": path.resolve(__dirname, './static/dist/js'),
    "filename": "bundle.js"
},
devServer: {
  contentBase: path.join(__dirname, './static/dist/js')
},
"module": {
  "rules": [ {
    "test": /\.css$/,
    "use": [ "style-loader", "css-loader" ]
  },
  {
    "test": /\.js$/,
    "exclude": /node_modules/,
    "use": {
      "loader": "babel-loader",
      "options": {
        "presets": [ "@babel/preset-env", ]
      }
    }
  }]
}}
