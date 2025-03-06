#!/bin/bash
# build-with-config.sh

# Run the normal build using the vite build command directly
npm run build:vite

# Create a backup of the original index.html
cp dist/index.html dist/index.html.bak

# Generate a new index.html with the embedded config
cat > dist/index.html << EOF
<!DOCTYPE html>
<html>
<head>
  <title>Wintermute</title>
  <script>
    window.VAPI_CONFIG = {
      "vapi_public_key": "${vapi_public_key}",
      "vapi_voice_id": "${VAPI_VOICE_ID}",
      "api_url": "${FRONTEND_URL}"
    };
  </script>
  <link rel="stylesheet" href="/assets/main-Ci3rcHx3.css">
</head>
<body>
  <div id="root"></div>
  <script type="module" src="/assets/vendor-D4I5kSlY.js"></script>
  <script type="module" src="/assets/main-Bx_lkUZU.js"></script>
</body>
</html>
EOF

echo "Custom index.html with embedded config created!"