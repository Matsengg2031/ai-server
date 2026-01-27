// Node.js Runtime (Default)

export default async function handler(req: Request) {
    return new Response(JSON.stringify({ status: "ok", message: "Server is alive" }), {
        status: 200,
        headers: { "Content-Type": "application/json" }
    });
}
"gemini-3-flash-preview",   // Worker 1
    "gemini-2.0-flash-001",     // Worker 2
    "gemini-2.5-flash-lite",    // Worker 3
    "gemini-2.5-pro"            // Judge (Tie-Breaker)
];

// Toggle ENSEMBLE MODE (true = gunakan voting dari semua model)
// TRUE = Judge Mode (2 Workers Paralel -> jika beda -> Judge decider)
const ENSEMBLE_MODE = true;

// Confidence threshold - di bawah ini = konservatif mode
const CONFIDENCE_THRESHOLD = 90;

const TTL_MS = 120 * 1000;

// ==================== STATE ====================
const recentAnswers = new Map<string, { answer: string; ts: number }>();
// In Vercel Edge, we cannot rely on background processing for long queues
// So we process "queue" immediately per request
// const inflight = new Map<string, Promise<{ answer: string; duration: number; confidence: number }>>();
// const requestQueue: Array<{...}> = []; 

// let isProcessing = false;
let requestCounter = 0;

// ==================== TYPES ====================
interface QuestionOption {
    label: string;
    text: string;
}

interface QuestionInput {
    question: string;
    options?: QuestionOption[];
    type?: string;
    number?: number;
    image?: string; // Base64 image
}

interface ModelResult {
    success: boolean;
    answer?: string;
    confidence?: number;
    model: string;
    raw?: string;
    error?: string;
}

interface VotingResult {
    finalAnswer: string;
    finalConfidence: number;
    votes?: Record<string, { count: number; avgConfidence: number }>;
    modelAnswers?: Array<{ model: string; answer: string; confidence: number }>;
    method: string;
}

// ==================== API KEY ====================
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
    console.error("❌ GEMINI_API_KEY tidak ditemukan.");
}

// GoogleGenAI initialization
const ai = new GoogleGenerativeAI(apiKey || "");

// ==================== UTILITIES ====================
function normalizePrompt(s = ""): string {
    return String(s)
        .replace(/\s+\n/g, "\n")
        .replace(/\n\s+/g, "\n")
        .replace(/[ \t]+/g, " ")
        .replace(/[.。…]+$/g, "")
        .trim();
}

function vacuum(): void {
    const now = Date.now();
    for (const [k, v] of recentAnswers) {
        if (now - v.ts > TTL_MS) recentAnswers.delete(k);
    }
}

// ==================== ERROR FORMATTER (USER-FRIENDLY) ====================
function formatError(error: Error | string): string {
    const msg = String((error as Error).message || error || "Unknown error");

    // Try to parse JSON error from API
    try {
        if (msg.includes('{"error"')) {
            const jsonMatch = msg.match(/\{"error".*\}/);
            if (jsonMatch) {
                const parsed = JSON.parse(jsonMatch[0]);
                const code = parsed.error?.code;
                const status = parsed.error?.status;

                if (code === 503 || status === "UNAVAILABLE") return "🔄 Server Sibuk";
                if (code === 429) return "⏳ Rate limit";
                if (code === 400 && msg.includes("API key expired")) return "🔑 API key expired";
                if (code === 400 && msg.includes("API_KEY_INVALID")) return "🔑 API key invalid";
                if (code === 401) return "🔑 API key salah";
                if (code === 404) return "❓ Model tidak ada";
            }
        }
    } catch (_) { /* ignore */ }

    // Fallback to keyword matching
    if (msg.includes("503") || msg.includes("overload") || msg.includes("UNAVAILABLE")) {
        return "🔄 Server sibuk";
    }
    if (msg.includes("429") || msg.includes("RESOURCE_EXHAUSTED") || msg.includes("quota")) {
        return "⏳ Rate limit";
    }
    if (msg.includes("timeout") || msg.includes("DEADLINE_EXCEEDED")) {
        return "⏰ Timeout";
    }
    if (msg.includes("API key expired")) {
        return "🔑 API key expired";
    }
    if (msg.includes("401") || msg.includes("API_KEY") || msg.includes("authentication")) {
        return "🔑 API key invalid";
    }
    if (msg.includes("404") || msg.includes("not found")) {
        return "❓ Model tidak ada";
    }
    if (msg.includes("SAFETY") || msg.includes("blocked")) {
        return "🚫 Diblokir safety";
    }
    if (msg.includes("ECONNREFUSED") || msg.includes("network")) {
        return "📡 Network error";
    }
    if (msg.includes("Empty response") || msg.includes("parse failure")) {
        return "📭 Jawaban kosong";
    }

    // Generic - ambil 30 karakter pertama
    return msg.substring(0, 30) + (msg.length > 30 ? "..." : "");
}

// ==================== PROMPT BUILDER (AI REASONING MODE) ====================
function buildPrompt(input: string | QuestionInput): string {
    if (typeof input === "string") {
        return `You are a MikroTik certification exam expert operating in "DEEP THINKING MODE".

PROTOCOL:
1.  **Analyze**: Read the question and every single option carefully.
2.  **Evaluate**: For EACH option, write a detailed explanation why it is correct or incorrect.
3.  **Reasoning**: You MUST provide thorough reasoning (at least 5-10 sentences). Do NOT be lazy.
4.  **Conclusion**: Final Answer must be determined only after analysis.

FORMAT:
Reasoning:
[Your detailed step-by-step analysis here...]

JSON:
{"answer": "A", "confidence": 95}
`;
    }

    if (typeof input === "object") {
        let prompt = `You are a MikroTik certification exam expert.\n\n`;

        prompt += `Goal: Provide the most accurate answer based on deep technical reasoning.\n\n`;

        prompt += `INSTRUCTIONS:\n`;
        prompt += `1. Analyze the Question: Identify key constraints (e.g., "invalid", "not", "public IP").\n`;
        prompt += `2. Analyze Options: Evaluate each option's technical validity.\n`;
        prompt += `3. Chain of Thought: Explain your step-by-step reasoning.\n`;
        prompt += `4. Final Output: Return the answer in strictly valid JSON.\n\n`;

        prompt += `Question: ${input.question}\n`;

        if (input.options && input.options.length > 0) {
            prompt += `Options:\n`;
            input.options.forEach(opt => {
                prompt += `${opt.label}. ${opt.text}\n`;
            });

            if (input.type === "checkbox") {
                prompt += `\nType: CHECKBOX (select ALL correct answers)\n`;
            } else if (input.type === "select") {
                prompt += `\nType: TRUE/FALSE\n`;
            } else {
                prompt += `\nType: SINGLE CHOICE\n`;
            }
        }

        prompt += `\nOutput Format:\n`;
        prompt += `First, write your "Reasoning: ..." block.\n`;
        prompt += `Then, write "JSON: {"answer": "...", "confidence": ...}" on a new line.\n`;

        if (input.image) {
            prompt += `\n[IMAGE DETECTED] An image is provided with this question. Analyze the image carefully.\n`;
        }

        return prompt;
    }
    return "";
}

// ==================== RESPONSE EXTRACTION ====================
function extractText(response: unknown): string | null {
    try {
        const resp = response as { text?: string | (() => string); candidates?: Array<{ content?: { parts?: Array<{ text?: string }> }; finishReason?: string; safetyRatings?: unknown }> };
        if (typeof resp.text === "string") return resp.text;
        if (typeof resp.text === "function") return resp.text();
        const candidate = resp?.candidates?.[0];
        if (candidate?.content?.parts?.[0]?.text) {
            return candidate.content.parts[0].text;
        }

        // Debug output if text is missing
        console.log(`⚠️ Empty Response Debug: FinishReason=${candidate?.finishReason}`);
        if (candidate?.safetyRatings) {
            console.log(`⚠️ Safety: ${JSON.stringify(candidate.safetyRatings)}`);
        }
    } catch (e) {
        console.log(`⚠️ Extract Error: ${(e as Error).message}`);
    }
    return null;
}

// ==================== PARSE ANSWER WITH CONFIDENCE ====================
function parseAnswerWithConfidence(rawText: string | null): { answer: string; confidence: number } {
    if (!rawText) return { answer: "", confidence: 50 };

    const text = rawText.trim();

    // Try to parse as JSON first
    try {
        // Remove markdown code blocks if present
        const jsonStr = text.replace(/```json\n?|\n?```/g, '').trim();
        const parsed = JSON.parse(jsonStr);
        return {
            answer: String(parsed.answer || "").toUpperCase(),
            confidence: Math.min(100, Math.max(0, parseInt(parsed.confidence) || 50))
        };
    } catch (_) { /* ignore */ }

    // Try to find JSON object using flexible pattern (handling potential newlines)
    const jsonMatch = text.match(/\{[\s\S]*?"answer"[\s\S]*?\}/);
    if (jsonMatch) {
        try {
            const jsonStr = jsonMatch[0];
            // Clean up potential markdown formatting specific to the match
            const cleanJson = jsonStr.replace(/```json|```/g, "").trim();
            const parsed = JSON.parse(cleanJson);
            return {
                answer: String(parsed.answer || "").toUpperCase(),
                confidence: Math.min(100, Math.max(0, parseInt(parsed.confidence) || 50))
            };
        } catch (_) { /* ignore */ }
    }

    // Handle TRUE/FALSE responses
    if (/^(true|false)$/i.test(text)) {
        return { answer: text.toLowerCase(), confidence: 85 };
    }

    // Fallback: extract ONLY valid answer letters (A-F) - not random letters
    // Look for patterns like "A", "A, B", "A,B,C" at the start or standalone
    const answerPattern = text.match(/^\s*([A-F])(?:\s*[,\s]\s*([A-F]))*\s*$/i);
    if (answerPattern) {
        const letters = text.toUpperCase().match(/[A-F]/g);
        if (letters && letters.length > 0 && letters.length <= 6) {
            return {
                answer: [...new Set(letters)].sort().join(", "),
                confidence: 70
            };
        }
    }

    // Fallback 2: Look for "Answer: A" or "**Answer**: B" pattern commonly used in reasoning
    const textAnswerMatch = text.match(/(?:answer|jawaban|option)\s*:?\s*(\*+)?\s*([A-F](?:,\s*[A-F])*)(?:\s|$|\*)/i);
    if (textAnswerMatch) {
        const letters = textAnswerMatch[2].toUpperCase().match(/[A-F]/g);
        if (letters) {
            return {
                answer: [...new Set(letters)].sort().join(", "),
                confidence: 65
            };
        }
    }

    // Last resort: if text is very short and contains only A-F letters
    if (text.length <= 10) {
        const letters = text.toUpperCase().match(/[A-F]/g);
        if (letters && letters.length > 0 && letters.length <= 4) {
            return {
                answer: [...new Set(letters)].sort().join(", "),
                confidence: 60
            };
        }
    }

    // Failed to parse - return empty (this model's result will be skipped)
    return { answer: "", confidence: 0 };
}

// ==================== SINGLE MODEL CALL ====================
// ==================== SINGLE MODEL CALL (WITH RETRY) ====================
async function callSingleModel(model: string, prompt: string, imageBase64?: string, retries = 3): Promise<ModelResult> {
    for (let attempt = 1; attempt <= retries; attempt++) {
        try {
            const genModel = ai.getGenerativeModel({ model });

            // Construct parts (Text + Optional Image)
            const parts: any[] = [{ text: prompt }];
            if (imageBase64) {
                // Clean header if present (data:image/jpeg;base64,...)
                const cleanData = imageBase64.replace(/^data:image\/\w+;base64,/, "");
                parts.push({
                    inlineData: {
                        mimeType: "image/jpeg",
                        data: cleanData
                    }
                });
            }

            const response = await genModel.generateContent({
                contents: [{ role: "user", parts: parts }],
                generationConfig: {
                    temperature: 0.4, // Seimbang
                    maxOutputTokens: 8192,
                },
                safetySettings: [
                    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE }
                ]
            });

            const rawText = extractText(response.response);
            if (rawText && rawText.trim() !== "") {
                const { answer, confidence } = parseAnswerWithConfidence(rawText);
                if (answer) {
                    return { success: true, answer, confidence, model, raw: rawText };
                }
            }

            // If empty, throw error to trigger retry
            throw new Error("Empty response or parse failure");

        } catch (err) {
            const isLastAttempt = attempt === retries;
            const errMsg = (err as Error).message;
            const isOverload = errMsg.includes("503") || errMsg.includes("429") || errMsg.includes("Overload");

            if (isLastAttempt) {
                return { success: false, model, error: isOverload ? "Server Overload (Max Retries)" : formatError(err as Error) };
            }

            // Wait before retry (exponential backoff: 1s, 2s, 4s...)
            const waitTime = isOverload ? 2000 * attempt : 1000 * attempt;
            const friendlyError = formatError(err as Error);
            console.log(`   ⚠️ ${model}: ${friendlyError} → Retry ${attempt + 1}/${retries} (${Math.round(waitTime / 1000)}s)`);
            await new Promise(r => setTimeout(r, waitTime));
        }
    }
    return { success: false, model, error: "Unknown error" };
}

// ==================== ANSWER NORMALIZATION ====================
function normalizeAnswer(answer: string | undefined): string[] {
    if (!answer) return [];

    const upperAnswer = answer.toUpperCase().trim();

    // Handle TRUE/FALSE jawaban (untuk soal dropdown true/false)
    if (upperAnswer === "TRUE" || upperAnswer === "FALSE") {
        return [upperAnswer.toLowerCase()]; // return as "true" or "false"
    }

    // Handle regular letter answers (A, B, C, etc.)
    const labels = upperAnswer
        .split(/[\s,]+/)
        .map(s => s.trim())
        .filter(s => /^[A-Z]$/.test(s));
    return [...new Set(labels)].sort();
}

// ==================== CONFIDENCE-WEIGHTED VOTING ====================
function _confidenceWeightedVote(modelResults: ModelResult[]): VotingResult {
    const successfulResults = modelResults.filter(r => r.success);

    if (successfulResults.length === 0) {
        throw new Error("All models failed in ensemble");
    }

    // Jika hanya 1 model berhasil → langsung pakai
    if (successfulResults.length === 1) {
        const r = successfulResults[0];
        return {
            finalAnswer: r.answer!,
            finalConfidence: r.confidence!,
            votes: { [r.answer!]: { count: 1, avgConfidence: r.confidence! } },
            modelAnswers: [{ model: r.model, answer: r.answer!, confidence: r.confidence! }],
            method: "single"
        };
    }

    // STEP 1: Penalize models that give too many answers (4+ = unreliable)
    const reliableResults = successfulResults.map(r => {
        const labels = normalizeAnswer(r.answer);
        if (labels.length >= 4) {
            // Model ini tidak reliable, turunkan confidence drastis
            return { ...r, confidence: Math.min(r.confidence!, 40), unreliable: true };
        }
        return { ...r, unreliable: false };
    });

    // STEP 2: Look for EXACT answer matches between models
    const answerCounts: Record<string, { count: number; totalConfidence: number; models: string[] }> = {};
    reliableResults.forEach(r => {
        const key = r.answer!.toUpperCase().trim();
        if (!answerCounts[key]) {
            answerCounts[key] = { count: 0, totalConfidence: 0, models: [] };
        }
        answerCounts[key].count++;
        answerCounts[key].totalConfidence += r.confidence!;
        answerCounts[key].models.push(r.model);
    });

    // Sort by count desc, then by avg confidence desc
    const sortedAnswers = Object.entries(answerCounts)
        .map(([answer, data]) => ({
            answer,
            count: data.count,
            avgConfidence: Math.round(data.totalConfidence / data.count),
            models: data.models
        }))
        .sort((a, b) => {
            if (b.count !== a.count) return b.count - a.count;
            return b.avgConfidence - a.avgConfidence;
        });

    let finalAnswer: string, finalConfidence: number, method: string;

    // STEP 3: Decision logic
    const topAnswer = sortedAnswers[0];

    if (topAnswer && topAnswer.count >= 2) {
        // Majority consensus (2+ models agree on EXACT answer)
        finalAnswer = topAnswer.answer;
        finalConfidence = topAnswer.avgConfidence;
        method = "consensus";
    } else {
        // No consensus → use PRIMARY model (first in MODELS list)
        const primaryResult = reliableResults.find(r => r.model === MODELS[0] && !r.unreliable);

        if (primaryResult) {
            finalAnswer = primaryResult.answer!;
            finalConfidence = primaryResult.confidence!;
            method = "primary";
        } else {
            // Primary model unreliable → use second model
            const secondaryResult = reliableResults.find(r => r.model === MODELS[1] && !r.unreliable);

            if (secondaryResult) {
                finalAnswer = secondaryResult.answer!;
                finalConfidence = secondaryResult.confidence!;
                method = "secondary";
            } else {
                // All unreliable → pick the one with highest confidence anyway
                const best = reliableResults.reduce((a, b) =>
                    a.confidence! > b.confidence! ? a : b
                );
                finalAnswer = best.answer!;
                finalConfidence = best.confidence!;
                method = "fallback";
            }
        }
    }

    // Build votes summary for logging
    const votes: Record<string, { count: number; avgConfidence: number }> = {};
    sortedAnswers.forEach(a => {
        votes[a.answer] = { count: a.count, avgConfidence: a.avgConfidence };
    });

    return {
        finalAnswer,
        finalConfidence,
        votes,
        modelAnswers: reliableResults.map(r => ({
            model: r.model + (r.unreliable ? " ⚠️" : ""),
            answer: r.answer!,
            confidence: r.confidence!
        })),
        method
    };
}

// ==================== ENSEMBLE CALL (JUDGE MODE) ====================
async function callGeminiEnsemble(prompt: string, options: QuestionOption[] = [], imageBase64?: string): Promise<VotingResult> {
    console.log(`🎯 Judge Mode: Running 3 Workers Parallel...`);

    // 1. Run Worker 1, Worker 2 & Worker 3 in parallel
    const workers = [MODELS[0], MODELS[1], MODELS[2]];
    const promises = workers.map(model => callSingleModel(model, prompt, imageBase64));
    const results = await Promise.all(promises);

    // Log Worker Results
    results.forEach(r => {
        if (r.success) console.log(`   ✓ ${r.model}: ${r.answer} (${r.confidence}%)`);
        else console.log(`   ✗ ${r.model}: ${r.error}`);
    });

    // Normalize answer to Label(s) - extract only A-F letters
    const normalize = (ans: string | undefined, opts: QuestionOption[]): string => {
        if (!ans) return "";
        const a = ans.trim().toUpperCase();

        // Handle TRUE/FALSE directly
        if (/^TRUE$/i.test(a)) return "true";
        if (/^FALSE$/i.test(a)) return "false";

        // For single letter A or B, check if this is a TRUE/FALSE question
        if (opts && opts.length === 2) {
            const optTexts = opts.map(o => o.text.toLowerCase().trim());
            const isTrueFalseQuestion = optTexts.includes("true") || optTexts.includes("false");

            if (isTrueFalseQuestion) {
                const trueOpt = opts.find(o => o.text.toLowerCase().trim() === "true");
                const falseOpt = opts.find(o => o.text.toLowerCase().trim() === "false");

                if (a === "A" && trueOpt && trueOpt.label === "A") return "true";
                if (a === "A" && falseOpt && falseOpt.label === "A") return "false";
                if (a === "B" && trueOpt && trueOpt.label === "B") return "true";
                if (a === "B" && falseOpt && falseOpt.label === "B") return "false";
            }
        }

        const letters = a.match(/[A-F]/g);
        if (letters && letters.length > 0) {
            return [...new Set(letters)].sort().join(", ");
        }

        if (opts && opts.length > 0) {
            const exact = opts.find(o => o.text && o.text.trim().toLowerCase() === a.toLowerCase());
            if (exact) return exact.label;
        }

        return a;
    };

    // Normalize all worker answers
    const normalizedAnswers = results.map(r => r.success ? normalize(r.answer, options) : null);

    // Count votes for each answer
    const voteCounts: Record<string, { count: number; totalConfidence: number; models: string[] }> = {};
    results.forEach((r, i) => {
        if (r.success && normalizedAnswers[i]) {
            const ans = normalizedAnswers[i]!;
            if (!voteCounts[ans]) {
                voteCounts[ans] = { count: 0, totalConfidence: 0, models: [] };
            }
            voteCounts[ans].count++;
            voteCounts[ans].totalConfidence += r.confidence!;
            voteCounts[ans].models.push(r.model);
        }
    });

    // Find majority answer (2+ votes out of 3)
    const sortedVotes = Object.entries(voteCounts)
        .map(([answer, data]) => ({
            answer,
            count: data.count,
            avgConfidence: Math.round(data.totalConfidence / data.count),
            models: data.models
        }))
        .sort((a, b) => {
            if (b.count !== a.count) return b.count - a.count;
            return b.avgConfidence - a.avgConfidence;
        });

    const topVote = sortedVotes[0];

    // 2. Check for majority (2+ workers agree)
    if (topVote && topVote.count >= 2) {
        console.log(`   🤝 Majority! ${topVote.count}/3 workers agree: ${topVote.answer} (${topVote.models.join(", ")})`);
        return {
            finalAnswer: topVote.answer,
            finalConfidence: topVote.avgConfidence,
            method: topVote.count === 3 ? "unanimous" : "majority"
        };
    }

    // 3. No majority (all 3 different or failures) -> Call JUDGE (Model 4)
    console.log(`   ⚖️  No majority. Calling JUDGE (${MODELS[3]})...`);

    const judgeResult = await callSingleModel(MODELS[3], prompt, imageBase64);

    if (judgeResult.success) {
        console.log(`   👨‍⚖️ Judge Decision: ${judgeResult.answer} (${judgeResult.confidence}%)`);
        return {
            finalAnswer: judgeResult.answer!,
            finalConfidence: judgeResult.confidence!,
            method: "judge"
        };
    } else {
        // If Judge fails, fallback to whichever worker succeeded with highest confidence
        console.log(`   ⚠️ Judge failed. Fallback to best worker.`);
        const best = results.filter(r => r.success).sort((a, b) => b.confidence! - a.confidence!)[0];
        if (best) {
            return {
                finalAnswer: best.answer!,
                finalConfidence: best.confidence!,
                method: "fallback_worker"
            };
        }
    }

    throw new Error("All models failed including Judge");
}

// ==================== FAILOVER CALL (Non-Ensemble) ====================
async function callGeminiWithFailover(prompt: string, imageBase64?: string): Promise<{ result: string; model: string; confidence: number }> {
    for (let i = 0; i < MODELS.length; i++) {
        const model = MODELS[i];
        const result = await callSingleModel(model, prompt, imageBase64);

        if (result.success) {
            return { result: result.answer!, model, confidence: result.confidence! };
        }

        console.log(`⚠️  ${model}: ${result.error}, trying next...`);
    }

    throw new Error("All models failed");
}

// ==================== QUEUE PROCESSOR ADAPTATION ====================
// In Vercel, we run this directly inside the handler (no background loop)
async function processRequestDirectly(key: string, rawPrompt: string | QuestionInput, startTime: number): Promise<{ answer: string; duration: number; confidence: number }> {
    requestCounter++;
    try {
        let finalAnswer: string, modelInfo: string, confidence: number;

        // 🤖 AI Fallback: Use Ensemble or Single Model
        const prompt = buildPrompt(rawPrompt);
        const options = (typeof rawPrompt === 'object' && rawPrompt.options) ? rawPrompt.options : [];
        const image = (typeof rawPrompt === 'object' && rawPrompt.image) ? rawPrompt.image : undefined;

        if (ENSEMBLE_MODE) {
            const voting = await callGeminiEnsemble(prompt, options, image);
            finalAnswer = voting.finalAnswer;
            confidence = voting.finalConfidence;
            modelInfo = `ENSEMBLE [${voting.method}]`;
        } else {
            const { result, model, confidence: conf } = await callGeminiWithFailover(prompt, image);
            finalAnswer = result;
            confidence = conf;
            modelInfo = model;
        }

        const duration = Date.now() - startTime;
        recentAnswers.set(key, { answer: finalAnswer, ts: Date.now() });

        const qNum = typeof rawPrompt === 'object' && rawPrompt.number ? rawPrompt.number + ". " : "";
        const qText = typeof rawPrompt === 'object' ? (rawPrompt.question ? rawPrompt.question.substring(0, 55) + "..." : "JSON Data") : rawPrompt;
        console.log(`📝 Soal     : ${qNum}${qText}`);
        console.log(`💡 Jawaban  : ${finalAnswer} [${confidence}%]`);
        console.log(`🤖 Model    : ${modelInfo}`);
        console.log(`⏱️  Waktu    : ${duration}ms`);
        console.log("");

        return { answer: finalAnswer, duration, confidence };

    } catch (err) {
        console.error(`❌ [#${requestCounter}] ${(err as Error).message}`);
        throw err;
    }
}

// ==================== CORS HELPER ====================
function corsHeaders(): Record<string, string> {
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "no-store, max-age=0",
    };
}

// ==================== REQUEST HANDLER ====================
export default async function handler(req: Request): Promise<Response> {
    const url = new URL(req.url);
    const path = url.pathname;
    const method = req.method;

    // Handle CORS preflight
    if (method === "OPTIONS") {
        return new Response(null, { status: 204, headers: corsHeaders() });
    }

    // ==================== ROUTES ====================

    // GET /health
    if (method === "GET" && (path === "/health" || path === "/")) {
        return new Response(JSON.stringify({
            ok: true,
            ensemble: ENSEMBLE_MODE,
            confidenceThreshold: CONFIDENCE_THRESHOLD,
            queue: 0,
            cache: recentAnswers.size,
            platform: "Vercel Edge"
        }), {
            status: 200,
            headers: { ...corsHeaders(), "Content-Type": "application/json" }
        });
    }

    // POST /ask
    if (method === "POST" && path === "/ask") {
        try {
            const body = await req.json();
            const input = body.prompt;

            if (!input) {
                return new Response(JSON.stringify({ error: "Prompt kosong." }), {
                    status: 400,
                    headers: { ...corsHeaders(), "Content-Type": "application/json" }
                });
            }

            let rawPrompt: string | QuestionInput, key: string;
            if (typeof input === "string") {
                rawPrompt = input.trim();
                key = normalizePrompt(rawPrompt);
            } else {
                rawPrompt = input as QuestionInput;
                key = normalizePrompt(input.question) + "_" + JSON.stringify(input.options);
            }

            vacuum();

            const cached = recentAnswers.get(key);
            if (cached && Date.now() - cached.ts <= TTL_MS) {
                return new Response(JSON.stringify({
                    answer: cached.answer,
                    responseTimeMs: 0,
                    cached: true
                }), {
                    status: 200,
                    headers: { ...corsHeaders(), "Content-Type": "application/json" }
                });
            }

            const startTime = Date.now();

            // DIRECT PROCESSING (No background queue in Serverless)
            const { answer, duration, confidence } = await processRequestDirectly(key, rawPrompt, startTime);

            return new Response(JSON.stringify({
                answer,
                responseTimeMs: duration,
                confidence,
                cached: false,
                ensemble: ENSEMBLE_MODE
            }), {
                status: 200,
                headers: { ...corsHeaders(), "Content-Type": "application/json" }
            });

        } catch (e) {
            return new Response(JSON.stringify({ error: (e as Error).message || "Invalid JSON" }), {
                status: 400,
                headers: { ...corsHeaders(), "Content-Type": "application/json" }
            });
        }
    }

    // 404 Not Found
    return new Response("Not Found", {
        status: 404,
        headers: { ...corsHeaders(), "Content-Type": "text/plain" }
    });
}
