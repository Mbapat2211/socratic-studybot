"use client";

import React, { useState, useRef } from "react";

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
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [response, setResponse] = useState<SocraticResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploaded = e.target.files;
    if (!uploaded) return;

    for (const file of Array.from(uploaded)) {
      const text = await file.text();
      setFiles((prev) => [
        ...prev,
        {
          id: Date.now().toString() + Math.random(),
          filename: file.name,
          text,
          enabled: true,
        },
      ]);
    }
    // Reset input so same file can be re-uploaded if needed
    e.target.value = "";
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
    <div style={{ display: "flex", height: "100vh", backgroundColor: "#020817", color: "#f8fafc", fontFamily: "system-ui, -apple-system, sans-serif" }}>

      {/* SIDEBAR */}
      <aside style={{ width: "300px", borderRight: "1px solid #1e293b", backgroundColor: "#0f172a", padding: "24px 16px", display: "flex", flexDirection: "column", justifyContent: "space-between", flexShrink: 0 }}>
        <div>
          {/* Title */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "24px" }}>
            <span style={{ fontSize: "20px" }}>⚡</span>
            <h1 style={{ fontWeight: "700", fontSize: "16px", margin: 0 }}>Study Workspace</h1>
          </div>

          {/* Provider badge */}
          <div style={{ display: "flex", alignItems: "center", gap: "10px", padding: "10px 12px", backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px", marginBottom: "24px" }}>
            <span style={{ fontSize: "18px" }}>⚡</span>
            <div>
              <p style={{ margin: 0, fontSize: "12px", fontWeight: "600", color: "#fbbf24" }}>ElevenLabs</p>
              <p style={{ margin: 0, fontSize: "11px", color: "#94a3b8" }}>Conversational AI · Text-only</p>
            </div>
          </div>

          {/* File Upload */}
          <div style={{ marginBottom: "16px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
              <span style={{ fontSize: "11px", fontWeight: "600", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Attached Context
              </span>
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{ fontSize: "12px", color: "#60a5fa", background: "none", border: "none", cursor: "pointer", padding: "2px 6px", borderRadius: "4px" }}
              >
                + Upload File
              </button>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".txt,.md,.pdf,.csv,.json"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
            </div>

            {files.length === 0 && (
              <div
                onClick={() => fileInputRef.current?.click()}
                style={{ border: "2px dashed #334155", borderRadius: "8px", padding: "20px", textAlign: "center", cursor: "pointer", color: "#475569", fontSize: "12px" }}
              >
                <div style={{ fontSize: "24px", marginBottom: "6px" }}>📄</div>
                Click to upload .txt, .md, .pdf files
              </div>
            )}

            <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "300px", overflowY: "auto" }}>
              {files.map((file) => (
                <div
                  key={file.id}
                  style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px", backgroundColor: "#1e293b", borderRadius: "6px", border: "1px solid #334155", fontSize: "12px" }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "8px", overflow: "hidden" }}>
                    <input
                      type="checkbox"
                      checked={file.enabled}
                      onChange={() => toggleFile(file.id)}
                      style={{ accentColor: "#3b82f6", flexShrink: 0 }}
                    />
                    <span style={{ fontSize: "14px" }}>📄</span>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", color: "#cbd5e1" }}>{file.filename}</span>
                  </div>
                  <button
                    onClick={() => deleteFile(file.id)}
                    style={{ background: "none", border: "none", cursor: "pointer", color: "#475569", fontSize: "16px", flexShrink: 0, padding: "0 2px" }}
                    title="Remove file"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div style={{ fontSize: "11px", color: "#475569", borderTop: "1px solid #1e293b", paddingTop: "12px" }}>
          Active Context:{" "}
          <span style={{ color: "#34d399" }}>{files.filter((f) => f.enabled).length} file(s)</span>
        </div>
      </aside>

      {/* MAIN */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "32px", overflowY: "auto" }}>
        <div style={{ maxWidth: "720px", width: "100%", margin: "0 auto" }}>
          <div style={{ borderBottom: "1px solid #1e293b", paddingBottom: "16px", marginBottom: "24px" }}>
            <h2 style={{ fontSize: "20px", fontWeight: "700", margin: "0 0 4px 0" }}>Socratic Cognitive Tutor</h2>
            <p style={{ fontSize: "13px", color: "#64748b", margin: 0 }}>Direct answers are restricted. Guidance targets offline execution.</p>
          </div>

          {!response && !loading && (
            <div style={{ textAlign: "center", color: "#334155", marginTop: "80px" }}>
              <div style={{ fontSize: "48px", marginBottom: "12px" }}>🎓</div>
              <p style={{ fontSize: "14px" }}>Upload your study materials and ask a question to get Socratic guidance.</p>
            </div>
          )}

          {loading && (
            <div style={{ textAlign: "center", color: "#64748b", marginTop: "80px" }}>
              <div style={{ fontSize: "32px", marginBottom: "12px" }}>⏳</div>
              <p>Generating scaffold...</p>
            </div>
          )}

          {response && (
            <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
              <div style={{ padding: "16px", backgroundColor: "#0f172a", borderLeft: "4px solid #3b82f6", borderRadius: "0 8px 8px 0", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }}>
                <h3 style={{ fontSize: "11px", fontWeight: "700", color: "#60a5fa", textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 8px 0" }}>1. Textual Focus</h3>
                <p style={{ fontSize: "14px", color: "#e2e8f0", margin: 0, lineHeight: "1.6" }}>{response.textual_focus}</p>
              </div>

              <div style={{ padding: "16px", backgroundColor: "#0f172a", borderLeft: "4px solid #a855f7", borderRadius: "0 8px 8px 0", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }}>
                <h3 style={{ fontSize: "11px", fontWeight: "700", color: "#c084fc", textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 8px 0" }}>2. Thinking Framework</h3>
                <p style={{ fontSize: "14px", color: "#e2e8f0", margin: 0, lineHeight: "1.6" }}>{response.thinking_framework}</p>
              </div>

              <div style={{ padding: "16px", backgroundColor: "#0f172a", borderLeft: "4px solid #10b981", borderRadius: "0 8px 8px 0", boxShadow: "0 2px 8px rgba(0,0,0,0.3)" }}>
                <h3 style={{ fontSize: "11px", fontWeight: "700", color: "#34d399", textTransform: "uppercase", letterSpacing: "0.1em", margin: "0 0 8px 0" }}>3. Offline Notebook Task</h3>
                <p style={{ fontSize: "14px", color: "#e2e8f0", margin: 0, lineHeight: "1.6", fontWeight: "600" }}>{response.offline_exercise}</p>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <div style={{ maxWidth: "720px", width: "100%", margin: "24px auto 0" }}>
          <form onSubmit={handleSubmit} style={{ display: "flex", gap: "8px" }}>
            <input
              type="text"
              placeholder="Ask a question about the active materials..."
              value={studentQuery}
              onChange={(e) => setStudentQuery(e.target.value)}
              style={{ flex: 1, backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px", padding: "12px 16px", fontSize: "14px", color: "#f1f5f9", outline: "none" }}
            />
            <button
              type="submit"
              disabled={loading}
              style={{ backgroundColor: loading ? "#78350f" : "#f59e0b", color: "#0f172a", fontWeight: "700", padding: "12px 24px", borderRadius: "8px", fontSize: "14px", border: "none", cursor: loading ? "not-allowed" : "pointer", display: "flex", alignItems: "center", gap: "6px", opacity: loading ? 0.6 : 1 }}
            >
              {loading ? "⏳ Thinking..." : "➤ Ask"}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}
