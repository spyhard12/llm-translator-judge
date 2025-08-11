# LLM Translation Judge

Evaluates English-to-Filipino translations using two approaches:  
1. **Prompt-Engineered Judge** – single-prompt evaluation.  
2. **Agentic LLM Judge** – step-by-step agent with memory and tool use.

## Files
- **prompt_engineered_judge_main.py** – Runs the simple prompt-based judge (0/1 per criterion, final score is bucketed into 1, 3, or 5).  
- **agentic_judge_main.py** – Runs the agentic judge (0–10 per criterion, final score 1–5, uses tools and memory).  
- **agent_tools.py** – Helper tools for the agentic judge: semantic similarity, medical term checking, and grammar/style checking.

## Usage
1. Add your API key to a `.env` file:
   ```bash
   OPENAI_API_KEY=your_api_key
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   python prompt_engineered_judge_main.py
   ```
   or
   ```bash
   python agentic_judge_main.py
   ```
