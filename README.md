# Resolve.AI - Customer Support Intelligence

Resolve.AI is a modern web interface demonstrating a production-grade customer support AI pipeline. It showcases an interactive simulation of deterministic rule-based triage combined with Large Language Model (LLM) intent classification and ChromaDB RAG retrieval. The project is designed with a clean, soft-neubrutalist aesthetic using HTML, CSS, and vanilla JavaScript.

## Features

*   **Interactive AI Agent Demo:** A live, simulated chat interface that processes user input using mock rules for toxicity, PII, hardware issues, and LLM intent routing. It visually demonstrates how a query is either auto-handled or escalated based on risk.[cite: 2]
*   **Dynamic Dashboard:** Visualizes evaluation metrics (like Baseline vs. Agent Accuracy and Escalation Precision/Recall) using Chart.js. Includes interactive doughnut and bar charts.[cite: 2]
*   **Deep Dive Analytics:** Provides detailed per-intent performance metrics (Precision, Recall, F1-Score) and LLM-as-a-Judge auto-evaluation results in responsive tables.[cite: 2]
*   **AI Frontier Insights:** Analyzes common failure modes (e.g., LLM Domain Bias, Sarcasm Blindness) and proposes actionable research solutions for future iterations.[cite: 2]
*   **Modern Aesthetic:** Implements a soft-neubrutalist design with a muted sage-green and cream color palette, pronounced border radii, and crisp, flat shadows for a clean user experience.[cite: 2]

## Technologies Used

*   **Frontend:** HTML5, CSS3, Vanilla JavaScript
*   **Typography:** [Plus Jakarta Sans](https://fonts.google.com/specimen/Plus+Jakarta+Sans) & [Fira Code](https://fonts.google.com/specimen/Fira+Code) via Google Fonts[cite: 2]
*   **Charting:** [Chart.js](https://www.chartjs.org/) (loaded via CDN)[cite: 2]

## Installation and Usage

This is a static front-end project. No complex build tools or server setup are required to view the interface.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Yesh6957/resolve-ai.git](https://github.com/Yesh6957/resolve-ai.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd resolve-ai
    ```
3.  **Open the file:**
    Open the `index.html`  file directly in your preferred web browser.

## File Structure

*   `index.html`: The main, single-page application file containing all HTML structure, CSS styling, and JavaScript logic.

## Project Architecture Details

The front-end simulates a backend architecture composed of several key steps[cite: 2]:
1.  **Deterministic Safety Layer:** Pre-LLM regex patterns scan for PII, extreme toxicity, or high-risk keywords to prevent hallucinations during critical escalations.[cite: 2]
2.  **LLM Intent Classification:** A simulated Codestral LLM categorizes input into one of 7 strict JSON intents (e.g., `software_troubleshooting`).[cite: 2]
3.  **ChromaDB RAG Retrieval:** If deemed safe, the system simulates embedding the text to query a local database of historical resolutions.[cite: 2]
4.  **Grounded Generation:** Generates a concise response grounded in retrieved examples.[cite: 2]

## About the Developer

Built by Yeshwanth J., an AI/ML & Full Stack Engineer currently pursuing a Master of Computer Applications (expected 2026). The project highlights an understanding of end-to-end LLM pipelines, from data extraction to strict JSON parsing and RAG vector retrieval.[cite: 2]

*   [Project GitHub](https://github.com/Yesh6957/resolve-ai)[cite: 2]
*   [Report Document](https://drive.google.com/file/d/1v0LRXOznDyZps75GSKUwK3qaBDU-sY1U/view?usp=drivesdk)[cite: 2]
*   [Portfolio](https://yesh-portfolio-dev.vercel.app/)[cite: 2]
