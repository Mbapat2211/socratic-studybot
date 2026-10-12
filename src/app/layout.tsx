import "./globals.css";

export const metadata = {
  title: "Socratic Studybot Workspace",
  description: "AI Scaffold Engine forcing offline paper-and-pencil learning",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100">{children}</body>
    </html>
  );
}