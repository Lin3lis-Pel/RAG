// The API's types. api-types.ts is generated from the backend's Pydantic schemas (`npm run types`); this file only
// gives the ones the page uses short names.

import type { components } from "./api-types";

type Schemas = components["schemas"];

export type Source = Schemas["Source"];
export type ToolCall = Schemas["ToolCall"];
export type RagDebug = Schemas["RagDebugOut"];
export type Usage = Schemas["Usage"];
export type StreamEvent = Schemas["StreamEvent"];
export type StoredConversation = Schemas["ConversationOut"];
export type ErrorResponse = Schemas["ErrorResponse"];
