// HIA/frontend/src/hooks/useTheme.js
import { useCallback, useEffect } from "react";
import { useLocalStorage } from "./useLocalStorage";

export function useTheme(defaultMode = false) {
	const [isDarkMode, setIsDarkMode] = useLocalStorage("darkMode", defaultMode);

	const toggleTheme = useCallback(() => {
		setIsDarkMode((prevMode) => !prevMode);
	}, [setIsDarkMode]);

	useEffect(() => {
		// Apply to both html and body for consistency with the inline script in index.html
		const htmlClassList = document.documentElement.classList;
		const bodyClassList = document.body.classList;

		if (isDarkMode) {
			htmlClassList.add("dark-mode");
			bodyClassList.add("dark-mode");
			document.documentElement.style.colorScheme = "dark";
		} else {
			htmlClassList.remove("dark-mode");
			bodyClassList.remove("dark-mode");
			document.documentElement.style.colorScheme = "light";
		}

		// Cleanup function
		return () => {
			bodyClassList.remove("dark-mode");
		};
	}, [isDarkMode]);

	return { isDarkMode, toggleTheme };
}
