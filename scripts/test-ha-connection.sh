#!/bin/bash
# Test script to verify Home Assistant connection from addon

set -e

echo "=========================================="
echo "Home Assistant Connection Test"
echo "=========================================="
echo ""

# Load environment variables
if [ -f .env ]; then
    source .env
    echo "✓ Loaded .env file"
else
    echo "⚠ No .env file found"
fi

echo ""
echo "Configuration:"
echo "  SUPERVISOR_URL: ${SUPERVISOR_URL:-not set}"
echo "  SUPERVISOR_TOKEN: ${SUPERVISOR_TOKEN:+***configured (${#SUPERVISOR_TOKEN} chars)***}"
echo ""

if [ -z "$SUPERVISOR_TOKEN" ]; then
    echo "❌ SUPERVISOR_TOKEN is not set"
    echo ""
    echo "To configure:"
    echo "  1. Create a long-lived access token in Home Assistant"
    echo "  2. Copy .env.example to .env"
    echo "  3. Set SUPERVISOR_URL and SUPERVISOR_TOKEN in .env"
    exit 1
fi

if [ -z "$SUPERVISOR_URL" ]; then
    echo "❌ SUPERVISOR_URL is not set"
    exit 1
fi

echo "Testing connection to Home Assistant..."
echo ""

# Test 1: Basic connectivity
echo "Test 1: Can we reach the host?"
if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${SUPERVISOR_URL}" > /dev/null 2>&1; then
    echo "  ✓ Host is reachable"
else
    echo "  ❌ Cannot reach ${SUPERVISOR_URL}"
    echo "     Check that the URL is correct and HA is running"
    exit 1
fi

# Test 2: API endpoint without auth
echo "Test 2: Can we access the API endpoint?"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${SUPERVISOR_URL}/api/" 2>/dev/null || echo "000")
echo "  Response code: ${STATUS_CODE}"

if [ "$STATUS_CODE" = "401" ]; then
    echo "  ✓ API endpoint exists (401 = needs auth, which is expected)"
elif [ "$STATUS_CODE" = "200" ]; then
    echo "  ✓ API endpoint accessible"
else
    echo "  ⚠ Unexpected status code: ${STATUS_CODE}"
fi

# Test 3: API endpoint with auth
echo "Test 3: Can we authenticate?"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
    -H "Content-Type: application/json" \
    "${SUPERVISOR_URL}/api/" 2>/dev/null || echo "000")
echo "  Response code: ${STATUS_CODE}"

if [ "$STATUS_CODE" = "200" ]; then
    echo "  ✓ Authentication successful!"
elif [ "$STATUS_CODE" = "401" ]; then
    echo "  ❌ Authentication failed - token may be invalid"
    exit 1
else
    echo "  ❌ Unexpected response: ${STATUS_CODE}"
    exit 1
fi

# Test 4: Can we fetch history?
echo "Test 4: Can we access history API?"
HISTORY_URL="${SUPERVISOR_URL}/api/history/period"
STATUS_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: Bearer ${SUPERVISOR_TOKEN}" \
    -H "Content-Type: application/json" \
    "${HISTORY_URL}" 2>/dev/null || echo "000")
echo "  Response code: ${STATUS_CODE}"

if [ "$STATUS_CODE" = "200" ]; then
    echo "  ✓ History API accessible"
else
    echo "  ⚠ History API returned: ${STATUS_CODE}"
fi

echo ""
echo "=========================================="
echo "✓ All tests passed!"
echo "=========================================="
echo ""
echo "Your configuration should work. Now test with the addon:"
echo "  docker compose down"
echo "  docker compose up -d"
echo "  docker compose logs -f ihp-ml-addon"
echo ""
echo "Then check status:"
echo "  curl http://localhost:5000/api/v1/status | jq"
