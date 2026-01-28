
// ==================== CONFIGURATION ====================
const MODELS = [
    "gemini-2.0-flash",         // Worker 1 (Fast & Smart)
    "gemini-1.5-flash",         // Worker 2 (Reliable)
    "gemini-1.5-pro",           // Judge (High Reasoning)
];

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const ENSEMBLE_MODE = true;
const CONFIDENCE_THRESHOLD = 90;
const TTL_MS = 120 * 1000;

// ==================== STATE ====================
const recentAnswers = new Map();

// ==================== ACCESS ENV SAFELY ====================
const apiKey = process.env.GEMINI_API_KEY;

// ==================== HELPER: CALL GEMINI REST API ====================
async function callGeminiREST(model, prompt, imageBase64) {
    if (!apiKey) return { text: "", error: "Missing API Key" };

    const url = `${API_BASE}/${model}:generateContent?key=${apiKey}`;

    // Construct Payload
    const parts = [{ text: prompt }];
    if (imageBase64) {
        // Remove header if exists
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
            return { text: "", error: `API Error ${response.status}: ${errText}` };
        }

        const data = await response.json();
        const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        return { text: text };

    } catch (e) {
        return { text: "", error: e.message };
    }
}

// ==================== LOGIC UTILS ====================
function normalizePrompt(s) {
    if (!s) return "";
    return String(s).replace(/\s+/g, " ").trim();
}

function vacuum() {
    const now = Date.now();
    for (const [k, v] of recentAnswers) {
        if (now - v.ts > TTL_MS) recentAnswers.delete(k);
    }
}

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

function buildPrompt(input) {
    if (typeof input === "string") return input; // Simplified

    let p = `You are a MikroTik exam expert. Analyze the question and image (if present).\n\n`;
    p += `Question: ${input.question}\n`;
    if (input.options) {
        input.options.forEach((opt) => {
            p += `${opt.label}. ${opt.text}\n`;
        });
    }
    p += `\nProvide the correct answer in JSON format:\n{"answer": "A", "confidence": 90}\n`;
    p += `Reasoning first, then JSON.\n`;
    return p;
}

// ==================== EXPORT HANDLER ====================
module.exports = async (req, res) => {
    // Enable CORS
    res.setHeader('Access-Control-Allow-Credentials', true);
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
    res.setHeader('Access-Control-Allow-Headers', 'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version');

    if (req.method === 'OPTIONS') {
        res.status(200).end();
        return;
    }

    if (req.method === 'GET') {
        res.status(200).json({ status: "alive", mode: "PURE_JS_REST" });
        return;
    }

    if (req.method === 'POST') {
        try {
            const input = req.body.prompt;
            if (!input) throw new Error("No prompt provided");

            // Cache Key
            const key = typeof input === "string" ? normalizePrompt(input) : normalizePrompt(input.question);
            vacuum();

            if (recentAnswers.has(key)) {
                const cached = recentAnswers.get(key);
                res.status(200).json({ ...cached, cached: true });
                return;
            }

            const promptText = buildPrompt(input);
            const image = (typeof input === 'object' && input.image) ? input.image : undefined;
            const startTime = Date.now();

            let finalResult = { answer: "", confidence: 0 };

            console.log(`🤖 Processing: ${key.substring(0, 30)}... [Image: ${!!image}]`);

            // 1. Call Model 1
            const r1 = await callGeminiREST(MODELS[0], promptText, image);
            const p1 = parseAnswerWithConfidence(r1.text);

            if (p1.confidence >= 85 || !ENSEMBLE_MODE) {
                finalResult = p1;
            } else {
                // 2. Call Model 2 (Pro) if low confidence
                const r2 = await callGeminiREST(MODELS[2], promptText, image);
                const p2 = parseAnswerWithConfidence(r2.text);

                finalResult = (p2.confidence > p1.confidence) ? p2 : p1;
            }

            if (!finalResult.answer) {
                res.status(500).json({ error: "No answer found", raw: r1.text });
                return;
            }

            const responseData = {
                answer: finalResult.answer,
                confidence: finalResult.confidence,
                responseTimeMs: Date.now() - startTime
            };

            recentAnswers.set(key, { answer: finalResult.answer, ts: Date.now() });

            res.status(200).json(responseData);

        } catch (e) {
            res.status(400).json({ error: e.message });
        }
        return;
    }

    res.status(404).json({ error: "Not Found" });
};
