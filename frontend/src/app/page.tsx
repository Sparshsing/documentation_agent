"use client";

import React, { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";

interface Index {
  name: string;
  description: string;
  source?: string;
  type?: string;
}

interface RetrievedNode {
  node_id: string;
  node_score: number;
  node_text: string;
  node_metadata: Record<string, any>;
}

export default function Home() {
  const [indexes, setIndexes] = useState<Index[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [responseText, setResponseText] = useState<string>("");
  const [sources, setSources] = useState<RetrievedNode[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [dark, setDark] = useState<boolean>(false);
  const [includeSources, setIncludeSources] = useState<boolean>(true);
  const [topK, setTopK] = useState<number>(5);
  // Retrieval options
  const [mode, setMode] = useState<string>("hybrid"); // 'vector' | 'keyword' | 'hybrid'
  const [rerank, setRerank] = useState<boolean>(true);

  const selectedIndexObj = indexes.find((idx) => idx.name === selectedIndex);

  // const txt = "## ABCD \n second line \n### zxczc \nhello, whats up?"
//   const txt = `To interact with files in a chat-like manner, you can first upload the file and then include it as part of the content sent to a model for generation.

// Here's a general approach:
// 1.  Upload the file using a client method like \`client.files.upload()\`. This will provide a reference to the uploaded file.
// 2.  Use this file reference in the \`contents\` argument when calling a content generation method, such as \`client.models.generate_content_stream()\`, alongside any text prompts.

// For example, to get a summary of a PDF document:

// \`\`\`python
// from google import genai
// client = genai.Client()

// # Upload the PDF file
// sample_pdf = client.files.upload(file="path/to/your/test.pdf")

// # Generate content using the uploaded PDF
// response = client.models.generate_content_stream(
//   model="gemini-2.0-flash",
//   contents=["Give me a summary of this document:", sample_pdf],
// )

// # Process the streamed response
// for chunk in response:
//   print(chunk.text)
// \`\`\`
// `;
const txt = `
# Python code example
\`\`\`python
from google import genai
client = genai.Client()

# Upload the PDF file
sample_pdf = client.files.upload(file="path/to/your/test.pdf")

# Generate content using the uploaded PDF
response = client.models.generate_content_stream(
  model="gemini-2.0-flash",
  contents=["Give me a summary of this document:", sample_pdf],
)

# Process the streamed response
for chunk in response:
  print(chunk.text)
\`\`\`
`;


  // Fetch indexes on mount
  useEffect(() => {
    fetch("http://localhost:8000/api/v1/indexes")
      .then((res) => res.json())
      .then((data) => {
        setIndexes(data.indexes ?? []);
        if (data.indexes?.length) setSelectedIndex(data.indexes[0].name);
      })
      .catch((err) => {
        console.error(err);
        setError("Failed to load indexes");
      });
  }, []);

  // Toggle dark mode on state change
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !selectedIndex) return;

    setLoading(true);
    setError(null);
    setResponseText("");
    setSources(null);

    try {
      const res = await fetch("http://localhost:8000/api/v1/query-index", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query,
          index: selectedIndex,
          top_k: topK,
          include_sources: includeSources,
          mode,
          rerank,
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const data = await res.json();
      setResponseText(data.response);
      setSources(data.sources ?? null);
    } catch (err: any) {
      console.error(err);
      setError(err.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  };


  // return (
  //     <div className="markdown">
  //     <ReactMarkdown>{txt}</ReactMarkdown>
  //     </div>
  // )

  return (
    <div className="min-h-screen bg-white dark:bg-neutral-950 text-neutral-900 dark:text-neutral-100 transition-colors px-4 sm:px-8 py-10 flex flex-col">
      {/* Header */}
      <header className="flex items-center justify-between max-w-4xl w-full mx-auto mb-8">
        <h1 className="text-2xl sm:text-3xl font-semibold">Documentation Agent</h1>
        <button
          onClick={() => setDark((d) => !d)}
          className="rounded-md px-3 py-2 border border-neutral-300 dark:border-neutral-700 hover:bg-neutral-100 dark:hover:bg-neutral-800 text-sm"
        >
          {dark ? "Light Mode" : "Dark Mode"}
        </button>
      </header>

      {/* Main */}
      <main className="flex-1 w-full max-w-4xl mx-auto flex flex-col gap-6">
        {/* Form */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          {/* Instruction */}
          <p className="text-xs text-neutral-500">Please select an index and retrieval options before running a search.</p>

          <div className="flex flex-col sm:flex-row gap-4">
            <select
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(e.target.value)}
              className="flex-1 p-2 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-sm"
            >
              {indexes.map((idx) => (
                <option key={idx.name} value={idx.name}>
                  {idx.name}
                </option>
              ))}
            </select>

            <input
              type="number"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              min={1}
              className="w-24 p-2 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-sm"
              title="Top K results"
            />

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={includeSources}
                onChange={(e) => setIncludeSources(e.target.checked)}
              />
              Include sources
            </label>
          </div>

          {/* Selected index source */}
          {selectedIndexObj?.source && (
            <p className="text-xs text-neutral-500 break-all">
              Source: {" "}
              <a
                href={selectedIndexObj.source}
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
                {selectedIndexObj.source}
              </a>
            </p>
          )}

          {/* Retrieval options */}
          <div className="flex flex-col sm:flex-row gap-4">
            <label className="flex items-center gap-2 text-sm">
              Mode:
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="p-2 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-sm"
              >
                <option value="vector">vector</option>
                <option value="keyword">keyword</option>
                <option value="hybrid">hybrid</option>
              </select>
            </label>

            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={rerank}
                onChange={(e) => setRerank(e.target.checked)}
              />
              Use rerank
            </label>
          </div>
          
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            placeholder="Enter your question..."
            className="p-3 rounded-md border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 text-sm"
            required
          />

          <button
            type="submit"
            disabled={loading}
            className="self-start bg-black dark:bg-white text-white dark:text-black px-4 py-2 rounded-md hover:opacity-90 disabled:opacity-50 text-sm"
          >
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="p-4 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md text-sm">
            {error}
          </div>
        )}

        {/* Response */}
        {responseText && (
          <section className="prose dark:prose-invert max-w-none">
            <div className="markdown">
              <ReactMarkdown>{responseText}</ReactMarkdown>
            </div>
          </section>
        )}

        {/* Sources */}
        {sources && sources.length > 0 && (
          <section className="mt-6">
            <h2 className="text-xl font-semibold mb-2">Sources</h2>
            <ul className="space-y-2">
              {sources?.map((node, idx) => (
                <li
                  key={node.node_id}
                  className="border border-neutral-300 dark:border-neutral-700 rounded-md overflow-hidden text-sm"
                >
                  <details>
                    <summary className="cursor-pointer select-none p-3 bg-neutral-50 dark:bg-neutral-800">
                      Node {idx + 1} — score {node.node_score.toFixed(3)}
                    </summary>
                    <div className="p-3 space-y-3">
                      {node.node_metadata && Object.keys(node.node_metadata).length > 0 && (
                        <div className="text-xs space-y-1">
                          {Object.entries(node.node_metadata).map(([key, value]) => (
                            <div key={key} className="flex gap-1">
                              <span className="font-semibold">{key}:</span>
                              <span className="break-all">{String(value)}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="prose dark:prose-invert max-w-none">
                        <div className="markdown">
                          <ReactMarkdown>{node.node_text}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  </details>
                </li>
              ))}
            </ul>
          </section>
        )}
      </main>

      <footer className="mt-12 text-xs text-center text-neutral-500">
        Powered by Documentation Agent
      </footer>
    </div>
  );
}
