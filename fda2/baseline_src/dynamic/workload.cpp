#include "workload.h"
#include <iostream>
#include <sstream>
#include <algorithm>

/**
 * Extract string value from JSON
 * Example: "type": "insert" -> returns "insert"
 */
std::string SimpleJSONParser::extractString(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return "";

    pos += searchKey.length();
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json[pos] != '"') return "";
    pos++; // Skip opening quote

    size_t endPos = json.find('"', pos);
    if (endPos == std::string::npos) return "";

    return json.substr(pos, endPos - pos);
}

/**
 * Extract unsigned integer from JSON
 * Example: "timestamp": 100 -> returns 100
 */
unsigned SimpleJSONParser::extractUnsigned(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\":";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return 0;

    pos += searchKey.length();
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    // Extract number
    std::string numStr;
    while (pos < json.length() && (isdigit(json[pos]) || json[pos] == '-')) {
        numStr += json[pos++];
    }

    return numStr.empty() ? 0 : std::stoul(numStr);
}

/**
 * Extract float array from JSON
 * Example: "vector": [1.0, 2.0, 3.0] -> returns {1.0, 2.0, 3.0}
 */
std::vector<float> SimpleJSONParser::extractFloatArray(const std::string& json, const std::string& key) {
    std::vector<float> result;

    std::string searchKey = "\"" + key + "\":";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return result;

    pos += searchKey.length();
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json[pos] != '[') return result;
    pos++; // Skip opening bracket

    // Find closing bracket
    size_t endPos = json.find(']', pos);
    if (endPos == std::string::npos) return result;

    std::string arrayContent = json.substr(pos, endPos - pos);

    // Parse floats separated by commas
    std::stringstream ss(arrayContent);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        if (!token.empty()) {
            result.push_back(std::stof(token));
        }
    }

    return result;
}

/**
 * Extract unsigned integer array from JSON
 * Example: "groundtruth": [123, 456, 789] -> returns {123, 456, 789}
 */
std::vector<unsigned> SimpleJSONParser::extractUnsignedArray(const std::string& json, const std::string& key) {
    std::vector<unsigned> result;

    std::string searchKey = "\"" + key + "\":";
    size_t pos = json.find(searchKey);
    if (pos == std::string::npos) return result;

    pos += searchKey.length();
    // Skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json[pos] != '[') return result;
    pos++; // Skip opening bracket

    // Find closing bracket
    size_t endPos = json.find(']', pos);
    if (endPos == std::string::npos) return result;

    std::string arrayContent = json.substr(pos, endPos - pos);

    // Parse unsigned ints separated by commas
    std::stringstream ss(arrayContent);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);

        if (!token.empty()) {
            result.push_back(std::stoul(token));
        }
    }

    return result;
}

/**
 * Parse single JSONL line
 */
WorkloadEvent SimpleJSONParser::parseEvent(const std::string& jsonLine) {
    WorkloadEvent event;

    // Parse type
    std::string typeStr = extractString(jsonLine, "type");

    // Support both naming conventions: "t"/"timestamp", "id"/"point_id", "vec"/"vector"
    if (typeStr == "insert") {
        event.type = EVENT_INSERT;
        event.pointId = extractUnsigned(jsonLine, "id");
        if (event.pointId == 0) event.pointId = extractUnsigned(jsonLine, "point_id");

        event.vector = extractFloatArray(jsonLine, "vec");
        if (event.vector.empty()) event.vector = extractFloatArray(jsonLine, "vector");

        event.timestamp = extractUnsigned(jsonLine, "t");
        if (event.timestamp == 0) event.timestamp = extractUnsigned(jsonLine, "timestamp");
    } else if (typeStr == "delete") {
        event.type = EVENT_DELETE;
        event.pointId = extractUnsigned(jsonLine, "id");
        if (event.pointId == 0) event.pointId = extractUnsigned(jsonLine, "point_id");

        event.timestamp = extractUnsigned(jsonLine, "t");
        if (event.timestamp == 0) event.timestamp = extractUnsigned(jsonLine, "timestamp");
    } else if (typeStr == "query") {
        event.type = EVENT_QUERY;
        event.queryId = extractUnsigned(jsonLine, "query_id");
        if (event.queryId == 0) {
            // Workload generator doesn't use query_id, extract from vec field
            event.queryId = 0; // We'll use the vector directly
        }

        event.vector = extractFloatArray(jsonLine, "vec");
        if (event.vector.empty()) event.vector = extractFloatArray(jsonLine, "vector");

        // Extract embedded groundtruth (from our Python generator)
        event.groundtruth = extractUnsignedArray(jsonLine, "groundtruth");

        event.timestamp = extractUnsigned(jsonLine, "t");
        if (event.timestamp == 0) event.timestamp = extractUnsigned(jsonLine, "timestamp");
    }

    return event;
}

/**
 * Constructor: Load workload from file
 */
Workload::Workload(const std::string& filename)
    : totalInserts(0), totalDeletes(0), totalQueries(0), scenario("unknown") {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open workload file: " << filename << std::endl;
        return;
    }

    std::string line;
    unsigned queriesWithGT = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // Try to extract scenario from first line if present
        if (scenario == "unknown" && line.find("\"scenario\"") != std::string::npos) {
            scenario = SimpleJSONParser::extractString(line, "scenario");
        }

        // Skip metadata lines
        if (line.find("\"type\":\"metadata\"") != std::string::npos) {
            continue;
        }

        // Parse event
        WorkloadEvent event = SimpleJSONParser::parseEvent(line);

        // Update counters
        if (event.type == EVENT_INSERT) {
            totalInserts++;
        } else if (event.type == EVENT_DELETE) {
            totalDeletes++;
        } else if (event.type == EVENT_QUERY) {
            totalQueries++;
            if (event.hasGroundtruth()) {
                queriesWithGT++;
            }
        }

        events.push_back(event);
    }

    file.close();

    // Debug: report embedded GT statistics
    std::cout << "[Workload Parser] Queries with embedded GT: " << queriesWithGT
              << " / " << totalQueries << std::endl;
}

/**
 * Print workload summary
 */
void Workload::printSummary() const {
    std::cout << "Workload Summary:\n";
    std::cout << "  Scenario: " << scenario << "\n";
    std::cout << "  Total events: " << events.size() << "\n";
    std::cout << "    Inserts: " << totalInserts
              << " (" << (100.0 * totalInserts / events.size()) << "%)\n";
    std::cout << "    Deletes: " << totalDeletes
              << " (" << (100.0 * totalDeletes / events.size()) << "%)\n";
    std::cout << "    Queries: " << totalQueries
              << " (" << (100.0 * totalQueries / events.size()) << "%)\n";
}
