// src/types/vapi.d.ts
declare module "@vapi-ai/web" {
    interface VapiOptions {
        apiKey: string;
        voiceId: string;
        webhookUrl: string;
    }

    interface AssistantResponseData {
        type: 'final-transcript' | 'partial-transcript' | 'audio' | string; // Include 'string' for other types
        message?: string;
        audio_url?: string;
    }

    class Vapi {
        constructor(options: VapiOptions);
        start(): Promise<void>;
        stop(): void;
        on(event: 'assistant-response', callback: (data: AssistantResponseData) => void): void;
        on(event: 'error', callback: (error: Error) => void): void;
        on(event: 'ready', callback: () => void): void;
        on(event: 'started', callback: () => void): void;
        on(event: 'ended', callback: () => void): void;
        on(event: string, callback: (...args: any[]) => void): void; // Generic event handler
    }

    export default Vapi;
}