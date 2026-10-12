"use client";

import React, { useState } from "react";
import { Trash2, FileText, Send, Layers, Zap } from "lucide-react";

interface WorkspaceFile {
  id: string;
  filename: string;
  text: string;
  enabled: boolean;
}

interface SocraticResponse {
  textual_focus: string;
  thinking_framework: string;
  offline_exercise: string;
}

export default function WorkspacePage() {
  const [studentQuery, setStudentQuery] = useState("");
  const [files, setFiles] = useState<WorkspaceFile[]>([
    {
      id: "1",
      filename: "Midsummer_Act_3.txt",
      text: "PUCK: My mistress with a monster is in love. Near to her close and consecrated bower, While she was in her dull and sleeping hour...",
      enabled: true,
    },
  ]);
  const [response, setResponse] = useState<SocraticResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const addMockFile = () => {
    const newFile: WorkspaceFile = {
      id: Date.now().toString(),
      filename: `Lecture_Notes_Unit_${files.length + 1}.txt`,
      text: "Renaissance drama, meter vs prose, trochaic tetrameter used for supernatural characters like Puck.",
      enabled: true,
    };
    setFiles([...files, newFile]);
  };

  const toggleFile = (id: string) => {
    setFiles(files.map((f) => (f.id === id ? { ...f, enabled: !f.enabled } : f)));
  };

  const deleteFile = (id: string) => {
    setFiles(files.filter((f) => f.id !== id));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentQuery.trim()) return;

    setLoading(true);
    setResponse(null);

    try {
      const res = await fetch("/api/scaffold", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          student_query: studentQuery,
          workspace_files: files,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to fetch response");
      }

      const data: SocraticResponse = await res.json();
      setResponse(data);
    } catch (err: any) {
      alert(`Error: ${err.message || "Check server logs."}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100">
      <aside className="w-80 border-r border-slate-800 bg-slate-900 p-4 flex flex-col justify-between">
        <div>
          <div className="flex items-center gap-2 mb-6">
            <Layers className="text-blue-400" />
            <h1 className="font-bold text-lg">Study Workspace</h1>
          </div>

          <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg mb-6">
            <Zap className="w-4 h-4 text-yellow-400 flex-shrink-0" />
            <div>
              <p className="text-xs font-semibold text-slate-200">ElevenLabs</p>
              <p className="text-xs text-slate-400">Conversational AI · Text-only mode</p>
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-xs font-semibold text-slate-400 uppercase">Attached Context</span>
              <button onClick={addMockFile} className="text-xs text-blue-400 hover:underline">
                + Add File
              </button>
            </div>

            <div className="space-y-2 max-h-64 overflow-y-auto">
              {files.map((file) => (
                <div
                  key={file.id}
                  className="flex items-center justify-between p-2 bg-slate-800 rounded border border-slate-700 text-xs"
                >
                  <div className="flex items-center gap-2 overflow-hidden">
                    <input
                      type="checkbox"
                      checked={file.enabled}
                      onChange={() => toggleFile(file.id)}
                      className="rounded accent-blue-500"
                    />
                    <FileText className="w-4 h-4 text-slate-400 flex-shrink-0" />
                    <span className="truncate">{file.filename}</span>
                  </div>
                  <button onClick={() => deleteFile(file.id)} className="text-slate-500 hover:text-red-400">
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="text-xs text-slate-500 border-t border-slate-800 pt-3">
          Active Context Cache:{" "}
          <span className="text-emerald-400">{files.filter((f) => f.enabled).length} Files</span>
        </div>
      </aside>

      <main className="flex-1 flex flex-col justify-between p-8">
        <div className="max-w-3xl w-full mx-auto space-y-6">
          <header className="border-b border-slate-800 pb-4">
            <h2 className="text-xl font-bold text-slate-100">Socratic Cognitive Tutor</h2>
            <p className="text-sm text-slate-400">
              Direct answers are restricted. Guidance targets offline execution.
            </p>
          </header>

          {response && (
            <div className="space-y-4">
              <div className="p-4 bg-slate-900 border-l-4 border-blue-500 rounded-r shadow-md">
                <h3 className="text-xs font-bold text-blue-400 uppercase tracking-wider mb-1">
                  1. Textual Focus
                </h3>
                <p className="text-sm text-slate-200">{response.textual_focus}</p>
              </div>

              <div className="p-4 bg-slate-900 border-l-4 border-purple-500 rounded-r shadow-md">
                <h3 className="text-xs font-bold text-purple-400 uppercase tracking-wider mb-1">
                  2. Thinking Framework
                </h3>
                <p className="text-sm text-slate-200">{response.thinking_framework}</p>
              </div>

              <div className="p-4 bg-slate-900 border-l-4 border-emerald-500 rounded-r shadow-md">
                <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-wider mb-1">
                  3. Offline Notebook Task
                </h3>
                <p className="text-sm text-slate-200 font-semibold">{response.offline_exercise}</p>
              </div>
            </div>
          )}
        </div>

        <div className="max-w-3xl w-full mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              type="text"
              placeholder="Ask a question about the active materials..."
              value={studentQuery}
              onChange={(e) => setStudentQuery(e.target.value)}
              className="flex-1 bg-slate-800 border border-slate-700 rounded-lg p-3 text-sm text-slate-100 focus:outline-none focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-yellow-500 hover:bg-yellow-400 text-slate-900 font-semibold px-5 py-3 rounded-lg text-sm flex items-center gap-2 disabled:opacity-50 transition-colors"
            >
              {loading ? "Scaffolding..." : <><Send className="w-4 h-4" /> Ask</>}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
