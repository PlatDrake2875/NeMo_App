import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// Make body visible now that styles are loaded
document.body.style.visibility = "visible";

ReactDOM.createRoot(document.getElementById("root")).render(
	<React.StrictMode>
		<App />
	</React.StrictMode>,
);
