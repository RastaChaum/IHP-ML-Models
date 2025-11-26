#!/bin/bash
# Test script to verify addon functionality
set -e

BASE_URL="http://localhost:5000"

echo "=========================================="
echo "Testing IHP ML Models Addon"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    
    echo -n "Testing ${name}... "
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "${BASE_URL}${endpoint}")
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "${BASE_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [ "$http_code" == "200" ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        echo "  Response: $(echo $body | jq -c '.' 2>/dev/null || echo $body)"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $http_code)"
        echo "  Response: $body"
        return 1
    fi
}

# Wait for service to be ready
echo "Waiting for addon to be ready..."
for i in {1..30}; do
    if curl -s "${BASE_URL}/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Addon is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Addon failed to start${NC}"
        exit 1
    fi
    sleep 1
done

echo ""

# Run tests
passed=0
failed=0

if test_endpoint "Health Check" "GET" "/health" ""; then ((passed++)); else ((failed++)); fi
echo ""

if test_endpoint "Status" "GET" "/api/v1/status" ""; then ((passed++)); else ((failed++)); fi
echo ""

if test_endpoint "Train with Fake Data" "POST" "/api/v1/train/fake" '{"num_samples": 50}'; then ((passed++)); else ((failed++)); fi
echo ""

if test_endpoint "Predict" "POST" "/api/v1/predict" '{
    "outdoor_temp": 5.0,
    "indoor_temp": 18.0,
    "target_temp": 21.0,
    "humidity": 65.0,
    "hour_of_day": 7,
    "day_of_week": 1
}'; then ((passed++)); else ((failed++)); fi
echo ""

if test_endpoint "List Models" "GET" "/api/v1/models" ""; then ((passed++)); else ((failed++)); fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}${passed}${NC}"
echo -e "Failed: ${RED}${failed}${NC}"
echo ""

if [ $failed -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
