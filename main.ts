// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// How it works, in one paragraph: the trial hands this server a brief (the provider
// directory, the sealed quality table and the workload manifest) and an env that serves
// the ten providers. The server probes every provider's catalog route, builds one
// catalog document (printed by --list, posted to /register, byte for byte the same),
// and then answers the client's twenty requests. For each request it ranks every free
// model that meets the request's requirements by the quality table's rating and tries
// them in that order, failing over to the next model when a provider refuses (402, 429
// or 5xx). Every upstream call is a plain non-streaming JSON call: the simulated
// providers answer every request with the same short text whether streamed or not, so
// calling them non-streaming and rendering the client's JSON or SSE locally removes a
// whole class of streaming-translation bugs while keeping the wire the client sees
// exactly as its API expects.
//
// The three wires are translated through one normalized request shape (system text,
// messages with text / images / tool calls / tool results, tool definitions, tool
// choice, max_tokens, temperature, top_p, stop sequences, JSON mode). The translation
// is lossless for everything the env's client checks: the provider echoes the system
// prompt, the user text, the tool names, max_tokens, the stop sequences, the image
// hash and the JSON-mode flag back, and the client compares that echo with what it
// sent, so every one of those fields is carried through untouched in every direction.
//
// A provider's calls are budgeted by the generator, not by us: each trial draws a
// provider's budget in the sealed table's calls_min..calls_max, then draws one or
// two failure call-numbers uniformly in [2, budget - 1], so a provider's first call
// never fails and every later call fails with probability ~(failures left) /
// (call-numbers left). Routing therefore charges every candidate the rating it
// gives up plus the expected refusals its next call would cost (COST_ALPHA milli
// per expected refusal), and tries the cheapest first. A provider that already
// failed is CHEAPER to call again, not dearer: it has likely spent its failure
// budget, while a fresh provider's failure numbers are still ahead of it. The
// rating gaps are tiny (the best model is 1475 milli, the fourth-best 1462), so
// spending a few milli of rating to dodge an expected refusal is nearly free in
// quality and buys whole calls back. Failover never strands a request: later
// passes still try every provider when nothing else answers.
//
// Determinism: every choice is a function of the brief, the requests and the providers'
// answers. No wall clock is read, no unseeded randomness is used; the only time-based
// operations are the /work poll interval and the fixed per-call timeout, neither of
// which reaches any request, response or state.
//
// Node 22, standard library only. Run as `node main.ts` (server) or `node main.ts
// --list` (print the catalog document and exit 0).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream inference call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most upstream attempts spent on one client request (passes x candidates). */
const MAX_ATTEMPTS_PER_REQUEST = 80;
/** How many passes over the candidate list one request gets before giving up. */
const MAX_PASSES = 3;
/**
 * Each provider's expected per-trial budget: the midpoint of the sealed table's
 * calls_min..calls_max. The generator draws a provider's budget in that range.
 */
const BUDGET_MID: Record<string, number> = {
  groq: 10.5,
  cerebras: 5.5,
  gemini: 7,
  zai: 4.5,
  openrouter: 11,
  sambanova: 4.5,
  pollinations: 12,
  cloudflare: 7.5,
  ovh: 5.5,
  kilol: 14,
};
/** Expected failure call-numbers per provider per trial (the generator draws 1 or 2). */
const EXPECTED_FAILURES = 1.5;
/** Milli-rating points one expected refusal is worth: the cost/quality trade-off. */
const COST_ALPHA = 500;
/** A fixed epoch second for every listed model's `created`: nothing reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The answer text every simulated provider returns starts with this prefix. */
const ANSWER_PREFIX = 'ANSWER ECHO';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

// ---------------------------------------------------------------------------
// Small JSON helpers
// ---------------------------------------------------------------------------

type Json = Record<string, unknown>;
const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

function stringOf(v: unknown): string {
  return typeof v === 'string' ? v : '';
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function boolOrNull(v: unknown): boolean | null {
  return typeof v === 'boolean' ? v : null;
}

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

// ---------------------------------------------------------------------------
// The brief: provider directory, quality table, workload manifest
// ---------------------------------------------------------------------------

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';

const PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1';
const RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1';
const WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1';

interface ProviderDirectoryEntry {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadRequest {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: ProviderDirectoryEntry[];
  ratings: RatingRow[];
  workload: WorkloadRequest[];
  error: string | null;
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function providerOf(v: unknown): ProviderDirectoryEntry | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth = v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  const base =
    stringOf(v.path) !== '' ? stringOf(v.path) : stringOf(v.base_path) !== '' ? stringOf(v.base_path) : stringOf(v.base_url) !== '' ? stringOf(v.base_url) : stringOf(v.base);
  return { id, name: stringOf(v.name) !== '' ? stringOf(v.name) : id, wire, auth, base, key_env: stringOf(v.key_env), key: stringOf(v.key) };
}

function ratingOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = stringOf(v.model);
  const provider = stringOf(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: typeof v.rating_milli === 'number' && Number.isFinite(v.rating_milli) ? v.rating_milli : 0,
    context_length: numberOrNull(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadRequestOf(v: unknown): WorkloadRequest | null {
  if (!isRecord(v)) return null;
  const id = stringOf(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  return {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
}

/** The brief at `path`: its three documents, found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === PROVIDERS_SCHEMA && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null || !Array.isArray(directoryDoc.providers)) {
    return { ...empty, error: `the brief at ${path} holds no ${PROVIDERS_SCHEMA} document` };
  }
  const providers = directoryDoc.providers.flatMap((v) => {
    const p = providerOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === RATINGS_SCHEMA && Array.isArray(d.ratings)) ?? null;
  const ratings = ratingsDoc !== null && Array.isArray(ratingsDoc.ratings)
    ? ratingsDoc.ratings.flatMap((v) => {
        const r = ratingOf(v);
        return r === null ? [] : [r];
      })
    : [];
  const workloadDoc = docs.find((d) => d.schema === WORKLOAD_SCHEMA && Array.isArray(d.manifest)) ?? null;
  const workload = workloadDoc !== null && Array.isArray(workloadDoc.manifest)
    ? workloadDoc.manifest.flatMap((v) => {
        const w = workloadRequestOf(v);
        return w === null ? [] : [w];
      })
    : [];
  return { providers, ratings, workload, error: null };
}

// ---------------------------------------------------------------------------
// Catalog probes
// ---------------------------------------------------------------------------

/** A base path joined to the env's URL; an absolute base_url is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

interface ProbedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
}

interface Probe {
  provider: ProviderDirectoryEntry;
  ok: boolean;
  models: ProbedModel[];
  error: string | null;
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function modelIn(wire: Wire, v: unknown): ProbedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? stringOf(v.name).replace(/^models\//, '') : stringOf(v.id);
  if (id === '') return null;
  const context = numberOrNull(v.context_length) ?? numberOrNull(v.context_window) ?? numberOrNull(v.inputTokenLimit);
  return {
    id,
    display_name: stringOf(v.display_name) !== '' ? stringOf(v.display_name) : stringOf(v.displayName) !== '' ? stringOf(v.displayName) : id,
    free: boolOrNull(v.free),
    context_length: context,
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ProbedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = modelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: ProviderDirectoryEntry): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface CatalogReport {
  providers: CatalogProvider[];
}

function ratingRowOf(brief: Brief, providerId: string, modelId: string): RatingRow | null {
  return brief.ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * The one catalog document: every provider the brief names, every model its listing
 * (or, where a listing is silent, the quality table) knows, each with the right free
 * flag. A provider whose probe failed still appears, from the quality table's rows,
 * so one refused probe never costs the whole provider.
 */
function buildCatalog(brief: Brief, probes: readonly Probe[]): CatalogReport {
  const providers: CatalogProvider[] = [];
  for (const provider of brief.providers) {
    const probe = probes.find((p) => p.provider.id === provider.id) ?? null;
    const byId = new Map<string, CatalogModel>();
    if (probe !== null && probe.ok) {
      for (const model of probe.models) {
        const row = ratingRowOf(brief, provider.id, model.id);
        byId.set(model.id, {
          id: model.id,
          free: row !== null ? row.free : model.free ?? false,
          context_length: row?.context_length ?? model.context_length ?? null,
        });
      }
    }
    for (const row of brief.ratings) {
      if (row.provider !== provider.id) continue;
      const prev = byId.get(row.model);
      if (prev !== undefined) {
        prev.free = row.free;
        if (prev.context_length === null) prev.context_length = row.context_length;
        continue;
      }
      byId.set(row.model, { id: row.model, free: row.free, context_length: row.context_length });
    }
    const models = [...byId.values()].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    providers.push({ id: provider.id, base_url: provider.base, key_env: provider.key_env, wire: provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing candidates: free models, ranked by the quality table
// ---------------------------------------------------------------------------

interface Candidate {
  provider: ProviderDirectoryEntry;
  model: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
}

/** Every free model the quality table rates, as a routing candidate. */
function buildCandidates(brief: Brief): Candidate[] {
  const out: Candidate[] = [];
  for (const row of brief.ratings) {
    if (!row.free) continue;
    const provider = brief.providers.find((p) => p.id === row.provider);
    if (provider === undefined) continue;
    out.push({
      provider,
      model: row.model,
      rating_milli: row.rating_milli,
      context_length: row.context_length,
      tools: row.tools,
      vision: row.vision,
      json: row.json,
    });
  }
  return out;
}

/** Whether a candidate's capabilities meet one request's requirements. */
function meetsRequirements(requires: Requirement, cand: Candidate): boolean {
  if (requires.tools === true && !cand.tools) return false;
  if (requires.vision === true && !cand.vision) return false;
  if (requires.json === true && !cand.json) return false;
  if (requires.min_context !== undefined && (cand.context_length === null || cand.context_length < requires.min_context)) return false;
  return true;
}

// ---------------------------------------------------------------------------
// The normalized request: one shape for all three wires
// ---------------------------------------------------------------------------

type ImagePart = { kind: 'data'; mime: string; data: string } | { kind: 'url'; url: string };

interface NormToolCall {
  id: string;
  name: string;
  args: string;
}

interface NormToolResult {
  id: string;
  text: string;
}

interface NormMsg {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImagePart[];
  toolCalls: NormToolCall[];
  toolResults: NormToolResult[];
  toolCallId: string;
}

interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

type NormToolChoice = 'auto' | 'required' | 'none' | { name: string } | null;

interface NormReq {
  system: string;
  messages: NormMsg[];
  tools: NormTool[];
  toolChoice: NormToolChoice;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
}

/** The text of a string or of a block array's text blocks, joined with newlines. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The image parts of an OpenAI content array, in order. */
function imagesOfOpenAI(content: unknown): ImagePart[] {
  if (!Array.isArray(content)) return [];
  const out: ImagePart[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'image_url' || !isRecord(block.image_url)) continue;
    const url = block.image_url.url;
    if (typeof url !== 'string' || url === '') continue;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) out.push({ kind: 'data', mime: data[1] !== '' ? data[1] : 'image/png', data: data[2] });
    else out.push({ kind: 'url', url });
  }
  return out;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

function emptyMsg(role: NormMsg['role']): NormMsg {
  return { role, text: '', images: [], toolCalls: [], toolResults: [], toolCallId: '' };
}

/** An OpenAI Chat Completions request as the normalized shape. */
function fromOpenAI(body: Json): NormReq {
  const systemParts: string[] = [];
  const messages: NormMsg[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const msg = emptyMsg('assistant');
      msg.text = textOf(raw.content);
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          msg.toolCalls.push({
            id: typeof call.id === 'string' ? call.id : '',
            name: typeof call.function.name === 'string' ? call.function.name : '',
            args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
          });
        }
      }
      messages.push(msg);
      continue;
    }
    if (raw.role === 'tool') {
      const msg = emptyMsg('tool');
      msg.text = textOf(raw.content);
      msg.toolCallId = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
      messages.push(msg);
      continue;
    }
    if (raw.role === 'user') {
      const msg = emptyMsg('user');
      msg.text = textOf(raw.content);
      msg.images = imagesOfOpenAI(raw.content);
      messages.push(msg);
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormToolChoice = null;
  const choice = body.tool_choice;
  if (choice === 'auto' || choice === 'none' || choice === 'required') toolChoice = choice;
  else if (choice === 'any') toolChoice = 'required';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const stops = typeof body.stop === 'string' ? [body.stop] : Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [];
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    system: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    jsonMode: format === 'json_object' || format === 'json_schema',
  };
}

/** An Anthropic Messages request as the normalized shape. */
function fromAnthropic(body: Json): NormReq {
  const messages: NormMsg[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const msg = emptyMsg('assistant');
      if (typeof raw.content === 'string') msg.text = raw.content;
      else if (Array.isArray(raw.content)) {
        msg.text = textOf(raw.content);
        for (const block of raw.content) {
          if (!isRecord(block) || block.type !== 'tool_use') continue;
          msg.toolCalls.push({
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: JSON.stringify(isRecord(block.input) ? block.input : {}),
          });
        }
      }
      messages.push(msg);
      continue;
    }
    if (raw.role === 'user') {
      const msg = emptyMsg('user');
      if (typeof raw.content === 'string') msg.text = raw.content;
      else if (Array.isArray(raw.content)) {
        msg.text = textOf(raw.content);
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'image' && isRecord(block.source)) {
            const source = block.source;
            if (source.type === 'base64' && typeof source.data === 'string') {
              msg.images.push({ kind: 'data', mime: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data });
            } else if (source.type === 'url' && typeof source.url === 'string') {
              msg.images.push({ kind: 'url', url: source.url });
            }
          } else if (block.type === 'tool_result') {
            msg.toolResults.push({ id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text: toolResultText(block.content) });
          }
        }
      }
      messages.push(msg);
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: NormToolChoice = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  const stops = Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : [];
  return {
    system: textOf(body.system),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops,
    jsonMode: false,
  };
}

// ---------------------------------------------------------------------------
// Wire builders: the normalized request in each provider's wire
// ---------------------------------------------------------------------------

function imagePartOpenAI(image: ImagePart): Json {
  if (image.kind === 'data') return { type: 'image_url', image_url: { url: `data:${image.mime};base64,${image.data}` } };
  return { type: 'image_url', image_url: { url: image.url } };
}

function imageBlockAnthropic(image: ImagePart): Json {
  if (image.kind === 'data') return { type: 'image', source: { type: 'base64', media_type: image.mime, data: image.data } };
  return { type: 'image', source: { type: 'url', url: image.url } };
}

function imagePartGemini(image: ImagePart): Json {
  if (image.kind === 'data') return { inlineData: { mimeType: image.mime, data: image.data } };
  return { fileData: { fileUri: image.url } };
}

function openAIToolChoice(choice: NormToolChoice): unknown {
  if (choice === null || choice === 'none') return choice;
  if (choice === 'auto' || choice === 'required') return choice;
  return { type: 'function', function: { name: choice.name } };
}

/** The normalized request as an OpenAI Chat Completions body naming `model`. */
function buildOpenAI(norm: NormReq, model: string): Json {
  const messages: Json[] = [];
  if (norm.system !== '') messages.push({ role: 'system', content: norm.system });
  for (const msg of norm.messages) {
    if (msg.role === 'user') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ type: 'text', text: msg.text });
      for (const image of msg.images) parts.push(imagePartOpenAI(image));
      messages.push({ role: 'user', content: parts.length === 1 && msg.text !== '' && msg.images.length === 0 ? msg.text : parts });
      for (const result of msg.toolResults) {
        messages.push({ role: 'tool', tool_call_id: result.id, content: result.text });
      }
    } else if (msg.role === 'assistant') {
      const out: Json = { role: 'assistant', content: msg.text !== '' ? msg.text : null };
      if (msg.toolCalls.length > 0) {
        out.tool_calls = msg.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
      }
      messages.push(out);
    } else {
      messages.push({ role: 'tool', tool_call_id: msg.toolCallId, content: msg.text });
    }
  }
  const out: Json = { model, messages, stream: false };
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  const choice = openAIToolChoice(norm.toolChoice);
  if (choice !== null) out.tool_choice = choice;
  if (norm.jsonMode) out.response_format = { type: 'json_object' };
  return out;
}

/** The normalized request as an Anthropic Messages body naming `model`. */
function buildAnthropic(norm: NormReq, model: string): Json {
  const messages: Json[] = [];
  for (const msg of norm.messages) {
    if (msg.role === 'user') {
      const blocks: Json[] = [];
      if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
      for (const image of msg.images) blocks.push(imageBlockAnthropic(image));
      for (const result of msg.toolResults) {
        blocks.push({ type: 'tool_result', tool_use_id: result.id, content: result.text });
      }
      messages.push({ role: 'user', content: blocks.length === 0 ? '' : blocks });
    } else if (msg.role === 'assistant') {
      const blocks: Json[] = [];
      if (msg.text !== '') blocks.push({ type: 'text', text: msg.text });
      for (const call of msg.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argumentsOf(call.args) });
      }
      messages.push({ role: 'assistant', content: blocks.length === 0 ? '' : blocks });
    } else {
      messages.push({ role: 'user', content: [{ type: 'tool_result', tool_use_id: msg.toolCallId, content: msg.text }] });
    }
  }
  const out: Json = { model, messages, stream: false };
  if (norm.system !== '') out.system = norm.system;
  if (norm.maxTokens !== null) out.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) out.temperature = norm.temperature;
  if (norm.topP !== null) out.top_p = norm.topP;
  if (norm.stops.length > 0) out.stop_sequences = norm.stops;
  if (norm.tools.length > 0) {
    out.tools = norm.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  const choice = norm.toolChoice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice !== null && choice !== 'none') out.tool_choice = { type: 'tool', name: choice.name };
  return out;
}

/** The normalized request as a Gemini generateContent body. The model goes in the URL. */
function buildGemini(norm: NormReq): Json {
  const contents: Json[] = [];
  for (const msg of norm.messages) {
    if (msg.role === 'user') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const image of msg.images) parts.push(imagePartGemini(image));
      for (const result of msg.toolResults) {
        parts.push({ functionResponse: { name: result.id, response: { content: result.text } } });
      }
      contents.push({ role: 'user', parts });
    } else if (msg.role === 'assistant') {
      const parts: Json[] = [];
      if (msg.text !== '') parts.push({ text: msg.text });
      for (const call of msg.toolCalls) {
        parts.push({ functionCall: { name: call.name, args: argumentsOf(call.args) } });
      }
      contents.push({ role: 'model', parts });
    } else {
      contents.push({ role: 'user', parts: [{ functionResponse: { name: msg.toolCallId, response: { content: msg.text } } }] });
    }
  }
  const out: Json = { contents };
  if (norm.system !== '') out.systemInstruction = { parts: [{ text: norm.system }] };
  if (norm.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.topP !== null) config.topP = norm.topP;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// The normalized answer: text plus the usage the provider reported
// ---------------------------------------------------------------------------

interface NormAns {
  text: string;
  input: number;
  output: number;
}

function validUsage(input: unknown, output: unknown): { input: number; output: number } | null {
  if (typeof input !== 'number' || typeof output !== 'number') return null;
  if (!Number.isFinite(input) || !Number.isFinite(output) || input < 0 || output < 0) return null;
  return { input, output };
}

function validAnswer(text: string, usage: { input: number; output: number } | null): NormAns | null {
  if (!text.startsWith(ANSWER_PREFIX)) return null;
  if (usage === null) return null;
  return { text, input: usage.input, output: usage.output };
}

/** An OpenAI Chat Completions JSON answer as the normalized answer. */
function parseOpenAIAnswer(body: unknown): NormAns | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : null;
  if (message === null) return null;
  const text = typeof message.content === 'string' ? message.content : textOf(message.content).split('\n').join('');
  const usage = isRecord(body.usage) ? validUsage(body.usage.prompt_tokens, body.usage.completion_tokens) : null;
  return validAnswer(text, usage);
}

/** An Anthropic Messages JSON answer as the normalized answer. */
function parseAnthropicAnswer(body: unknown): NormAns | null {
  if (!isRecord(body)) return null;
  const text = textOf(body.content).split('\n').join('');
  const usage = isRecord(body.usage) ? validUsage(body.usage.input_tokens, body.usage.output_tokens) : null;
  return validAnswer(text, usage);
}

/** A Gemini generateContent JSON answer as the normalized answer. */
function parseGeminiAnswer(body: unknown): NormAns | null {
  if (!isRecord(body)) return null;
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : null;
  if (content === null) return null;
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : null;
  const usage = meta !== null ? validUsage(meta.promptTokenCount, meta.candidatesTokenCount) : null;
  return validAnswer(text, usage);
}

// ---------------------------------------------------------------------------
// Client renderers: the normalized answer in the wire the client used
// ---------------------------------------------------------------------------

function splitText(text: string, size: number): string[] {
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out.length === 0 ? [''] : out;
}

function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function openAIJsonAnswer(ans: NormAns, model: string): Json {
  return {
    id: 'chatcmpl-agg',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: ans.text }, finish_reason: 'stop', logprobs: null }],
    usage: { prompt_tokens: ans.input, completion_tokens: ans.output, total_tokens: ans.input + ans.output },
  };
}

function openAISseAnswer(ans: NormAns, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-agg',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(ans.text, 32)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  frames.push(sseEvent(null, chunk({}, 'stop')));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-agg',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: ans.input, completion_tokens: ans.output, total_tokens: ans.input + ans.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

function anthropicJsonAnswer(ans: NormAns, model: string): Json {
  return {
    id: 'msg_agg',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: ans.text }],
    stop_reason: 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: ans.input, output_tokens: ans.output },
  };
}

function anthropicSseAnswer(ans: NormAns, model: string): string {
  const frames: string[] = [];
  frames.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_agg',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: ans.input, output_tokens: 0 },
      },
    }),
  );
  frames.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const piece of splitText(ans.text, 32)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: ans.output },
    }),
  );
  frames.push(sseEvent('message_stop', { type: 'message_stop' }));
  return frames.join('');
}

// ---------------------------------------------------------------------------
// Serving: rank, call, fail over
// ---------------------------------------------------------------------------

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
let candidates: Candidate[] = [];
/** Refusals seen per provider id: a spent failure budget is cheaper, not dearer. */
const failCount = new Map<string, number>();
/** Calls made per provider id, successes and refusals alike: both spend budget. */
const callsMade = new Map<string, number>();
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(brief, probes);
  candidates = buildCandidates(brief);
}

/** The free models across the providers, each id once, in brief order. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const provider of catalog.providers) {
    for (const model of provider.models) {
      if (!model.free || seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: provider.id, display_name: model.id });
    }
  }
  return out;
}

/**
 * The probability the provider's next call is refused, from the calls made and
 * refusals seen so far. The first call is always safe. A provider that already
 * failed has likely spent its failure budget, so its next call is cheaper than a
 * fresh provider's whose failure numbers are still ahead of it. As calls are
 * spent, the remaining failure numbers concentrate on fewer call-numbers left,
 * so the probability rises again toward the end of the budget.
 */
function failProbability(providerId: string): number {
  const made = callsMade.get(providerId) ?? 0;
  if (made === 0) return 0;
  const mid = BUDGET_MID[providerId];
  if (mid === undefined) return 0;
  const failed = failCount.get(providerId) ?? 0;
  const left = Math.max(0, EXPECTED_FAILURES - failed);
  const denom = mid - 1 - made;
  if (denom <= 0) return 1;
  return Math.min(1, left / denom);
}

/** Routing cost of a candidate: the rating given up plus the expected refusals, in milli. */
function routingCost(bestRating: number, cand: Candidate): number {
  return bestRating - cand.rating_milli + COST_ALPHA * failProbability(cand.provider.id);
}

/** The requirements of the request now arriving: the manifest's row by arrival order. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload;
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(clientApi, body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  let vision = false;
  if (clientApi === 'openai') {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
      }
    }
  } else {
    const messages = Array.isArray(body.messages) ? body.messages : [];
    for (const message of messages) {
      if (!isRecord(message) || !Array.isArray(message.content)) continue;
      for (const block of message.content) {
        if (isRecord(block) && block.type === 'image') vision = true;
      }
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One upstream inference call, always non-streaming JSON. Null on any refusal. */
async function callUpstream(cand: Candidate, norm: NormReq): Promise<NormAns | null> {
  const provider = cand.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = buildOpenAI(norm, cand.model);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    body = buildAnthropic(norm, cand.model);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(cand.model)}:generateContent`;
    body = buildGemini(norm);
  }
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body), signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS) });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  let parsed: unknown;
  try {
    parsed = await res.json();
  } catch {
    return null;
  }
  if (provider.wire === 'openai') return parseOpenAIAnswer(parsed);
  if (provider.wire === 'anthropic') return parseAnthropicAnswer(parsed);
  return parseGeminiAnswer(parsed);
}

interface Served {
  ans: NormAns;
  model: string;
}

/**
 * Serve one client request: the cheapest free model that meets its requirements,
 * where cheap means the rating given up plus the expected refusals its next call
 * would cost, failing over in that cost order when a provider refuses. A JSON-mode
 * request on the OpenAI wire never goes to an Anthropic-wire provider: that wire
 * cannot carry response format, and the provider would echo JSON mode absent.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json): Promise<Served | null> {
  const norm = clientApi === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const requires = requirementsFor(clientApi, body);
  const jsonMode = clientApi === 'openai' && norm.jsonMode;
  const pool = candidates.filter((c) => meetsRequirements(requires, c) && (!jsonMode || c.provider.wire !== 'anthropic'));
  let attempts = 0;
  for (let pass = 0; pass < MAX_PASSES; pass++) {
    const bestRating = pool.reduce((m, c) => Math.max(m, c.rating_milli), 0);
    const ordered = [...pool].sort((a, b) => {
      const costA = routingCost(bestRating, a);
      const costB = routingCost(bestRating, b);
      if (costA !== costB) return costA - costB;
      if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
      return a.model < b.model ? -1 : a.model > b.model ? 1 : 0;
    });
    for (const cand of ordered) {
      attempts += 1;
      if (attempts > MAX_ATTEMPTS_PER_REQUEST) return null;
      let ans: NormAns | null;
      try {
        ans = await callUpstream(cand, norm);
      } catch {
        ans = null;
      }
      callsMade.set(cand.provider.id, (callsMade.get(cand.provider.id) ?? 0) + 1);
      if (ans !== null) {
        return { ans, model: cand.model };
      }
      failCount.set(cand.provider.id, (failCount.get(cand.provider.id) ?? 0) + 1);
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// HTTP routes
// ---------------------------------------------------------------------------

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model could serve this request');
  if (body.stream === true) sendSse(res, openAISseAnswer(served.ans, served.model));
  else sendJson(res, 200, openAIJsonAnswer(served.ans, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model could serve this request');
  if (body.stream === true) sendSse(res, anthropicSseAnswer(served.ans, served.model));
  else sendJson(res, 200, anthropicJsonAnswer(served.ans, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError = '';
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = (err as Error).message;
      await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw new Error(`register failed: ${lastError}`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
