import "../styles.css";

export const metadata = {
  title: "CareMind",
  description: "Medical document AI assistant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
