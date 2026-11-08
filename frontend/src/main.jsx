import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App.jsx";
import ErrorBoundary from "./ErrorBoundary.jsx";
import { startTheme } from "./theme.js";
import "./observatory.css";

startTheme();
createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <ErrorBoundary label="Model Run" global>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
