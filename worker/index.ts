import type { Message } from 'ai';
import { google } from '@ai-sdk/google';
import { streamText } from 'ai';

type JsonBody = {
  id: string;
  messages: Message[];
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    switch (url.pathname) {
      case "/api/chat": {
        const { messages } = await request.json<JsonBody>();
        const userMessage = messages[messages.length - 1]?.content || "";

        // 1. Vectorize search for relevant docs
        const vectorResults = await env.VECTORIZE.query(userMessage, {});
        // Extract text/snippets from your vector results
        const retrievedDocs = vectorResults.matches
          .map(match => match.metadata?.text || match.text || "")
          .filter(Boolean)
          .join('\n\n');

        // 2. Add retrieved context as a system message for RAG
        const ragMessages: Message[] = [
          {
            role: "system",
            content: `Use the following context for retrieval-augmented generation (RAG):\n${retrievedDocs}`
          },
          ...messages
        ];

        // 3. Call Gemini with RAG context
        const model = google('gemini-2.5-pro', {
          useSearchGrounding: true
        });
        const result = streamText({ model, messages: ragMessages });
        return result.toDataStreamResponse();
      }
      default: {
        return new Response(null, { status: 404 });
      }
    }
  },
} satisfies ExportedHandler<Env>;
