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
      <body>{children}</body>
    </html>
  );
}