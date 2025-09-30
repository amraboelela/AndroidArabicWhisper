#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <stdexcept>

// Transcription data structures
struct Word {
    std::string text;
    double start;
    double end;
    double probability;

    Word(const std::string& t, double s, double e, double p)
        : text(t), start(s), end(e), probability(p) {}
};

struct Segment {
    std::string text;
    double start;
    double end;
    double avg_logprob;
    std::vector<Word> words;

    Segment() = default;
    Segment(const std::string& t, double s, double e, double logprob)
        : text(t), start(s), end(e), avg_logprob(logprob) {}
};

struct TranscriptionResult {
    bool success;
    std::string language;
    double language_probability;
    double duration;
    std::vector<Segment> segments;
    std::string error;

    TranscriptionResult() : success(false), language_probability(0.0), duration(0.0) {}
};

class WhisperTranscriber {
private:
    std::string pythonPath;
    std::string modelPath;

    std::string executeCommandWithOutput(const std::string& command) {
        std::string result;
        char buffer[128];

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("popen() failed!");
        }

        while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
            result += buffer;
        }

        pclose(pipe);
        return result;
    }

    TranscriptionResult parseTranscriptionJson(const std::string& jsonStr) {
        TranscriptionResult result;

        // Basic JSON parsing - in a real implementation, you'd use a proper JSON library
        if (jsonStr.find("\"success\": true") != std::string::npos) {
            result.success = true;

            // Extract language
            size_t langStart = jsonStr.find("\"language\": \"");
            if (langStart != std::string::npos) {
                langStart += 13;
                size_t langEnd = jsonStr.find("\"", langStart);
                result.language = jsonStr.substr(langStart, langEnd - langStart);
            }

            // Extract language probability
            size_t probStart = jsonStr.find("\"language_probability\": ");
            if (probStart != std::string::npos) {
                probStart += 24;
                size_t probEnd = jsonStr.find(",", probStart);
                if (probEnd == std::string::npos) probEnd = jsonStr.find("}", probStart);
                result.language_probability = std::stod(jsonStr.substr(probStart, probEnd - probStart));
            }

            // Extract duration
            size_t durStart = jsonStr.find("\"duration\": ");
            if (durStart != std::string::npos) {
                durStart += 12;
                size_t durEnd = jsonStr.find(",", durStart);
                if (durEnd == std::string::npos) durEnd = jsonStr.find("}", durStart);
                result.duration = std::stod(jsonStr.substr(durStart, durEnd - durStart));
            }

            // Parse segments
            parseSegments(jsonStr, result);

        } else {
            result.success = false;
            size_t errorStart = jsonStr.find("\"error\": \"");
            if (errorStart != std::string::npos) {
                errorStart += 10;
                size_t errorEnd = jsonStr.find("\"", errorStart);
                result.error = jsonStr.substr(errorStart, errorEnd - errorStart);
            } else {
                result.error = "Unknown error in transcription";
            }
        }

        return result;
    }

    void parseSegments(const std::string& jsonStr, TranscriptionResult& result) {
        size_t segmentsStart = jsonStr.find("\"segments\": [");
        if (segmentsStart == std::string::npos) return;

        size_t pos = segmentsStart;
        while ((pos = jsonStr.find("{\"text\":", pos)) != std::string::npos) {
            Segment segment;

            // Extract text
            size_t textStart = jsonStr.find("\"text\": \"", pos);
            if (textStart != std::string::npos) {
                textStart += 9;
                size_t textEnd = jsonStr.find("\", \"start\":", textStart);
                if (textEnd == std::string::npos) textEnd = jsonStr.find("\",", textStart);
                segment.text = jsonStr.substr(textStart, textEnd - textStart);
            }

            // Extract start time
            size_t startPos = jsonStr.find("\"start\": ", pos);
            if (startPos != std::string::npos) {
                startPos += 9;
                size_t startEnd = jsonStr.find(",", startPos);
                segment.start = std::stod(jsonStr.substr(startPos, startEnd - startPos));
            }

            // Extract end time
            size_t endPos = jsonStr.find("\"end\": ", startPos);
            if (endPos != std::string::npos) {
                endPos += 7;
                size_t endEnd = jsonStr.find(",", endPos);
                segment.end = std::stod(jsonStr.substr(endPos, endEnd - endPos));
            }

            // Extract avg_logprob
            size_t logprobPos = jsonStr.find("\"avg_logprob\": ", endPos);
            if (logprobPos != std::string::npos) {
                logprobPos += 15;
                size_t logprobEnd = jsonStr.find(",", logprobPos);
                if (logprobEnd == std::string::npos) logprobEnd = jsonStr.find("}", logprobPos);
                segment.avg_logprob = std::stod(jsonStr.substr(logprobPos, logprobEnd - logprobPos));
            }

            // Parse words (simplified)
            parseWordsInSegment(jsonStr, pos, segment);

            result.segments.push_back(segment);
            pos = jsonStr.find("}", pos) + 1;
        }
    }

    void parseWordsInSegment(const std::string& jsonStr, size_t segmentStart, Segment& segment) {
        size_t wordsStart = jsonStr.find("\"words\": [", segmentStart);
        if (wordsStart == std::string::npos) return;

        size_t pos = wordsStart;
        while ((pos = jsonStr.find("{\"word\":", pos)) != std::string::npos) {
            size_t wordStart = jsonStr.find("\"word\": \"", pos);
            if (wordStart == std::string::npos) break;

            wordStart += 9;
            size_t wordEnd = jsonStr.find("\",", wordStart);
            std::string wordText = jsonStr.substr(wordStart, wordEnd - wordStart);

            // Extract word start time
            size_t startPos = jsonStr.find("\"start\": ", wordEnd);
            if (startPos == std::string::npos) break;
            startPos += 9;
            size_t startEnd = jsonStr.find(",", startPos);
            double start = std::stod(jsonStr.substr(startPos, startEnd - startPos));

            // Extract word end time
            size_t endPos = jsonStr.find("\"end\": ", startEnd);
            if (endPos == std::string::npos) break;
            endPos += 7;
            size_t endEnd = jsonStr.find(",", endPos);
            double end = std::stod(jsonStr.substr(endPos, endEnd - endPos));

            // Extract probability
            size_t probPos = jsonStr.find("\"probability\": ", endEnd);
            if (probPos == std::string::npos) break;
            probPos += 15;
            size_t probEnd = jsonStr.find("}", probPos);
            double probability = std::stod(jsonStr.substr(probPos, probEnd - probPos));

            segment.words.emplace_back(wordText, start, end, probability);

            pos = probEnd + 1;
            // Break if we've reached the end of words array
            if (jsonStr.find("]", pos) < jsonStr.find("{\"word\":", pos)) break;
        }
    }

public:
    WhisperTranscriber(const std::string& modelPath, const std::string& pythonPath = "python3")
        : pythonPath(pythonPath), modelPath(modelPath) {}

    TranscriptionResult transcribe(const std::string& audioFile) {
        try {
            std::string command = pythonPath + " whisper_model_caller.py \"" + modelPath + "\" \"" + audioFile + "\"";
            std::string jsonOutput = executeCommandWithOutput(command);

            return parseTranscriptionJson(jsonOutput);

        } catch (const std::exception& e) {
            TranscriptionResult result;
            result.success = false;
            result.error = "Error executing transcription: " + std::string(e.what());
            return result;
        }
    }
};