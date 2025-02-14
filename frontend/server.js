// frontend/server.js
const express = require('express');
const path = require('path');
const app = express();

const port = process.env.PORT || 3000;

// Serve static files from the 'dist' directory (where Vite builds)
app.use(express.static(path.join(__dirname, '../dist')));

// For all other requests, serve the index.html file.  This handles routing
// within your React application.  VERY IMPORTANT for single-page applications.
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../dist', 'index.html'));
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});