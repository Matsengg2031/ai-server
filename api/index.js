
// ==================== CONFIGURATION (STRICT RESTORE) ====================
// Models restored exactly as in server.ts
const MODELS = [
    "gemini-3-flash-preview",   // Worker 1
    "gemini-2.0-flash-001",     // Worker 2
    "gemini-2.5-flash-lite",    // Worker 3
    "gemini-2.5-pro"            // Judge (Tie-Breaker)
];

const EXACT_MODELS = MODELS;

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const ENSEMBLE_MODE = true;
const TTL_MS = 120 * 1000;

// ==================== STATE ====================
const recentAnswers = new Map();

// ==================== ACCESS ENV SAFELY ====================
const apiKey = process.env.GEMINI_API_KEY;

// ==================== HELPER: CALL GEMINI REST API ====================
async function callGeminiREST(model, prompt, imageBase64) {
    if (!apiKey) return { success: false, model, error: "Missing API Key" };

    const url = `${API_BASE}/${model}:generateContent?key=${apiKey}`;

    // Construct Payload
    const parts = [{ text: prompt }];
    if (imageBase64) {
        const cleanData = imageBase64.replace(/^data:image\/\w+;base64,/, "");
        parts.push({
            inline_data: {
                mime_type: "image/jpeg",
                data: cleanData
            }
        });
    }

    const payload = {
        contents: [{ parts: parts }],
        generationConfig: {
            temperature: 0.4,
            maxOutputTokens: 8192
        }
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errText = await response.text();
            // Handle Model Not Found by falling back?
            return { success: false, model, error: `API Error ${response.status}: ${errText}` };
        }

        const data = await response.json();
        const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        return { success: true, answer: text, confidence: 0, model, raw: text }; // Confidence parsed later

    } catch (e) {
        if (e.name === 'AbortError') return { success: false, model, error: "Timeout (15s limit)" };
        return { success: false, model, error: e.message };
    }
}

// ==================== LOGIC: PARSE ANSWER ====================
function parseAnswerWithConfidence(rawText) {
    if (!rawText) return { answer: "", confidence: 50 };
    const text = rawText.trim();

    // Try JSON
    try {
        const jsonMatch = text.match(/\{[\s\S]*?"answer"[\s\S]*?\}/);
        if (jsonMatch) {
            const cleanJson = jsonMatch[0].replace(/```json|```/g, "").trim();
            const parsed = JSON.parse(cleanJson);
            return {
                answer: String(parsed.answer || "").toUpperCase(),
                confidence: parseInt(parsed.confidence) || 70
            };
        }
    } catch (_) { }

    // Fallback Patterns
    const letters = text.toUpperCase().match(/[A-F]/g);
    if (letters && letters.length > 0) {
        return { answer: [...new Set(letters)].join(", "), confidence: 60 };
    }

    return { answer: "", confidence: 0 };
}

// ==================== LOGIC: NORMALIZE ANSWER ====================
function normalizeAnswer(answer) {
    if (!answer) return [];
    const upperAnswer = answer.toUpperCase().trim();
    if (upperAnswer === "TRUE" || upperAnswer === "FALSE") return [upperAnswer.toLowerCase()];
    const labels = upperAnswer.split(/[\s,]+/).map(s => s.trim()).filter(s => /^[A-Z]$/.test(s));
    return [...new Set(labels)].sort();
}

// ==================== LOGIC: PROMPT BUILDER ====================
function buildPrompt(input) {
    if (typeof input === "string") return input;

    let p = `You are a MikroTik certification exam expert operating in "DEEP THINKING MODE".\n\n`;
    p += `PROTOCOL:\n`;
    p += `1. Analyze: Read the question and every single option carefully (and image if provided).\n`;
    p += `2. Evaluate: For EACH option, write a detailed explanation.\n`;
    p += `3. Conclusion: Final Answer must be determined only after analysis.\n\n`;

    p += `Question: ${input.question}\n`;
    if (input.options) {
        input.options.forEach((opt) => {
            p += `${opt.label}. ${opt.text}\n`;
        });
    }

    p += `\nOutput Format:\n`;
    p += `Reasoning:\n[Your detailed analysis...]\n\n`;
    p += `JSON:\n{"answer": "A", "confidence": 95}\n`;
    return p;
}

// ==================== LOGIC: VOTING ALGORITHM ====================
function confidenceWeightedVote(modelResults) {
    const successful = modelResults.filter(r => r.success);
    if (successful.length === 0) return null;

    // Parse confidences
    const parsedResults = successful.map(r => {
        const p = parseAnswerWithConfidence(r.raw);
        return { ...r, answer: p.answer, confidence: p.confidence };
    }).filter(r => r.answer); // Must have valid answer

    if (parsedResults.length === 0) return null;

    // Explicit Majority Vote
    const counts = {};
    parsedResults.forEach(r => {
        if (!counts[r.answer]) counts[r.answer] = 0;
        counts[r.answer]++;
    });

    // Find Winner
    const sortedAnswers = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
    const topAnswer = sortedAnswers[0];

    if (counts[topAnswer] >= 2) {
        const supporting = parsedResults.filter(r => r.answer === topAnswer);
        const avgConf = supporting.reduce((sum, r) => sum + r.confidence, 0) / supporting.length;
        return {
            finalAnswer: topAnswer,
            finalConfidence: avgConf,
            method: "majority",
            models: supporting.map(s => s.model)
        };
    }

    // No Majority -> Return Primary (Model 0)
    const primary = parsedResults.find(r => r.model === EXACT_MODELS[0]);
    if (primary) {
        return { finalAnswer: primary.answer, finalConfidence: primary.confidence, method: "primary", models: [primary.model] };
    }

    // Fallback -> Highest Confidence
    const best = parsedResults.sort((a, b) => b.confidence - a.confidence)[0];
    return { finalAnswer: best.answer, finalConfidence: best.confidence, method: "fallback_confidence", models: [best.model] };
}

// ==================== LOGIC: ENSEMBLE RUNNER ====================
async function runEnsemble(prompt, imageBase64) {
    console.log("🎯 Running Ensemble Logic...");

    // 1. Run 3 Workers
    const workers = [EXACT_MODELS[0], EXACT_MODELS[1], EXACT_MODELS[2]];
    const promises = workers.map(m => callGeminiREST(m, prompt, imageBase64));
    const results = await Promise.all(promises);

    const vote = confidenceWeightedVote(results);

    // 2. If Majority, Return
    if (vote && vote.method === "majority") {
        console.log(`✅ Majority Consensus: ${vote.finalAnswer}`);
        return vote;
    }

    // 3. Else, Call Judge
    console.log("⚖️ No majority. Calling Judge...", EXACT_MODELS[3]);
    const judgeResult = await callGeminiREST(EXACT_MODELS[3], prompt, imageBase64);

    if (judgeResult.success) {
        const p = parseAnswerWithConfidence(judgeResult.raw);
        if (p.answer) {
            console.log(`👨‍⚖️ Judge Decided: ${p.answer}`);
            return { finalAnswer: p.answer, finalConfidence: p.confidence, method: "judge" };
        }
    }

    // 4. Judge Failed? Return best worker result
    if (vote) return vote;

    throw new Error("Ensemble failed completely.");
}

// ==================== HANDLER ====================
module.exports = async (req, res) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') { res.status(200).end(); return; }
    if (req.method === 'GET') { res.status(200).json({ status: "alive", mode: "FULL_ENSEMBLE_RESTORED" }); return; }

    if (req.method === 'POST') {
        try {
            const input = req.body.prompt;
            if (!input) throw new Error("No prompt");

            // Cache Logic
            const key = typeof input === "string" ? input : JSON.stringify(input);
            if (recentAnswers.has(key)) { // Simple cache
                res.status(200).json({ ...recentAnswers.get(key), cached: true });
                return;
            }

            const promptText = buildPrompt(input);
            const image = (typeof input === 'object' && input.image) ? input.image : undefined;
            const startTime = Date.now();

            const result = await runEnsemble(promptText, image);

            const responseData = {
                answer: result.finalAnswer,
                confidence: result.finalConfidence,
                responseTimeMs: Date.now() - startTime
            };

            recentAnswers.set(key, responseData);
            res.status(200).json(responseData);

        } catch (e) {
            res.status(500).json({ error: e.message });
        }
        return;
    }
    res.status(404).json({ error: "Not Found" });
};
